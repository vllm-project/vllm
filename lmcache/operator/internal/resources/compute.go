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

import (
	"encoding/json"
	"fmt"
	"math"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// ComputeResources returns resource requirements, either from overrides or auto-computed from L1 size.
func ComputeResources(spec *lmcachev1alpha1.LMCacheEngineSpec) corev1.ResourceRequirements {
	if spec.ResourceOverrides != nil {
		return *spec.ResourceOverrides
	}

	// memoryRequest = ceil(l1.sizeGB + 5) Gi
	memReqGi := int64(math.Ceil(spec.L1.SizeGB + 5))
	// memoryLimit = ceil(memoryRequest * 1.5) Gi
	memLimGi := int64(math.Ceil(float64(memReqGi) * 1.5))

	return corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("4"),
			corev1.ResourceMemory: resource.MustParse(fmt.Sprintf("%dGi", memReqGi)),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse(fmt.Sprintf("%dGi", memLimGi)),
		},
	}
}

// BuildContainerArgs maps CRD spec fields to server.py CLI flags.
func BuildContainerArgs(spec *lmcachev1alpha1.LMCacheEngineSpec) []string {
	args := []string{
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", derefInt32(getServerPort(spec), 5555)),
		"--http-port", fmt.Sprintf("%d", getHTTPPort(spec)),
		"--l1-size-gb", fmt.Sprintf("%.1f", spec.L1.SizeGB),
		"--chunk-size", fmt.Sprintf("%d", getChunkSize(spec)),
		"--max-workers", fmt.Sprintf("%d", getMaxWorkers(spec)),
		"--hash-algorithm", getHashAlgorithm(spec),
	}

	// Eviction args
	evPolicy := "LRU"
	evTrigger := 0.8
	evRatio := 0.2
	if spec.Eviction != nil {
		evPolicy = derefString(spec.Eviction.Policy, "LRU")
		evTrigger = derefFloat64(spec.Eviction.TriggerWatermark, 0.8)
		evRatio = derefFloat64(spec.Eviction.EvictionRatio, 0.2)
	}
	args = append(args,
		"--eviction-policy", evPolicy,
		"--eviction-trigger-watermark", fmt.Sprintf("%.2f", evTrigger),
		"--eviction-ratio", fmt.Sprintf("%.2f", evRatio),
	)

	// Prometheus args
	promEnabled := true
	if spec.Prometheus != nil {
		promEnabled = derefBool(spec.Prometheus.Enabled, true)
	}
	if !promEnabled {
		args = append(args, "--disable-prometheus")
	} else {
		promPort := int32(9090)
		if spec.Prometheus != nil {
			promPort = derefInt32(spec.Prometheus.Port, 9090)
		}
		args = append(args, "--prometheus-port", fmt.Sprintf("%d", promPort))
	}

	// L2 backend
	if spec.L2Backend != nil {
		l2JSON := buildL2AdapterJSON(spec.L2Backend)
		if l2JSON != "" {
			args = append(args, "--l2-adapter", l2JSON)
		}

		// L2 policies
		args = append(args,
			"--l2-store-policy", derefString(spec.L2Backend.StorePolicy, "default"),
			"--l2-prefetch-policy", derefString(spec.L2Backend.PrefetchPolicy, "default"),
			"--l2-prefetch-max-in-flight", fmt.Sprintf("%d", derefInt32(spec.L2Backend.PrefetchMaxInFlight, 8)),
		)
	}

	// Coordinator registration. The URL is resolved by the controller (from a
	// coordinator ref or an explicit url) before this builder runs; an empty URL
	// means registration is disabled. The advertise IP falls back to the pod IP
	// via the LMCACHE_COORDINATOR_ADVERTISE_IP env var injected by the DaemonSet
	// builder, so it is only emitted as a flag when explicitly set.
	if c := spec.Coordinator; c != nil {
		if url := derefString(c.URL, ""); url != "" {
			args = append(args, "--coordinator-url", url)
			if c.AdvertiseIP != nil && *c.AdvertiseIP != "" {
				args = append(args, "--coordinator-advertise-ip", *c.AdvertiseIP)
			}
			args = append(args,
				"--coordinator-heartbeat-interval", formatFloat(derefFloat64(c.HeartbeatInterval, 5.0)),
			)
			if derefBool(c.L2EventReporting, false) {
				args = append(args, "--coordinator-l2-event-reporting")
			}
			args = append(args,
				"--coordinator-l2-event-flush-interval", formatFloat(derefFloat64(c.L2EventFlushInterval, 1.0)),
			)
		}
	}

	// User-supplied extra args (appended last so they can override defaults)
	args = append(args, spec.ExtraArgs...)

	return args
}

// buildL2AdapterJSON serializes an L2BackendSpec into the --l2-adapter JSON string.
// For RESP adapters with authSecretRef, username/password are set to env var
// placeholders that get interpolated by the shell wrapper (see BuildShellCommand).
// When a serde is configured, a "serde" sub-dict is attached to the adapter
// config so the server wraps the adapter with it.
func buildL2AdapterJSON(backend *lmcachev1alpha1.L2BackendSpec) string {
	var flat map[string]any
	switch {
	case backend.RESP != nil:
		flat = buildRESPL2Config(backend.RESP)
	case backend.Raw != nil:
		flat = buildRawL2Config(backend.Raw)
	default:
		return ""
	}
	if serde := buildSerdeConfig(backend.Serde); serde != nil {
		flat["serde"] = serde
	}
	b, err := json.Marshal(flat)
	if err != nil {
		return ""
	}
	return string(b)
}

func buildRESPL2Config(resp *lmcachev1alpha1.RESPL2AdapterSpec) map[string]any {
	flat := map[string]any{
		"type":        "resp",
		"host":        resp.Host,
		"port":        resp.Port,
		"num_workers": derefInt32(resp.NumWorkers, 8),
	}
	if resp.MaxCapacityGB != nil && *resp.MaxCapacityGB > 0 {
		flat["max_capacity_gb"] = *resp.MaxCapacityGB
	}
	// Auth credentials are passed via LMCACHE_RESP_USERNAME /
	// LMCACHE_RESP_PASSWORD env vars (injected by the DaemonSet builder),
	// not in the JSON config.
	return flat
}

func buildRawL2Config(raw *lmcachev1alpha1.RawL2AdapterSpec) map[string]any {
	flat := make(map[string]any)
	flat["type"] = raw.Type
	for k, v := range raw.Config {
		var parsed any
		if err := json.Unmarshal(v.Raw, &parsed); err != nil {
			flat[k] = string(v.Raw)
		} else {
			flat[k] = parsed
		}
	}
	return flat
}

// buildSerdeConfig renders L2SerdeSpec into the serde sub-dict consumed by
// the server's adapter factory, or nil when no serde is configured. For
// aesgcm, master_key_path points at the file mounted by the DaemonSet
// builder from the managed master-key Secret.
func buildSerdeConfig(serde *lmcachev1alpha1.L2SerdeSpec) map[string]any {
	if serde == nil || serde.AESGCM == nil {
		return nil
	}
	return map[string]any{
		"type":            "aesgcm",
		"key_provider":    derefString(serde.AESGCM.KeyProvider, "hkdf"),
		"master_key_path": l2EncryptionKeyPath,
		"aes_bits":        derefInt32(serde.AESGCM.AESBits, 128),
	}
}

func getServerPort(spec *lmcachev1alpha1.LMCacheEngineSpec) *int32 {
	if spec.Server != nil {
		return spec.Server.Port
	}
	return nil
}

func getHTTPPort(spec *lmcachev1alpha1.LMCacheEngineSpec) int32 {
	if spec.Server != nil {
		return derefInt32(spec.Server.HTTPPort, 8080)
	}
	return 8080
}

func getChunkSize(spec *lmcachev1alpha1.LMCacheEngineSpec) int32 {
	if spec.Server != nil {
		return derefInt32(spec.Server.ChunkSize, 256)
	}
	return 256
}

func getMaxWorkers(spec *lmcachev1alpha1.LMCacheEngineSpec) int32 {
	if spec.Server != nil {
		return derefInt32(spec.Server.MaxWorkers, 1)
	}
	return 1
}

func getHashAlgorithm(spec *lmcachev1alpha1.LMCacheEngineSpec) string {
	if spec.Server != nil {
		return derefString(spec.Server.HashAlgorithm, "blake3")
	}
	return "blake3"
}
