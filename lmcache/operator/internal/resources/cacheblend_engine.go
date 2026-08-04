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
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	// cbEngineType is the value of the --engine-type flag that selects the
	// CacheBlend V3 engine on the lmcache server binary. The server maps the
	// value "blend" to BlendV3Module; "blend_v3" is no longer recognized
	// (lmcache/v1/multiprocess/server.py).
	cbEngineType = "blend"

	// cbL1AlignBytes is the value of the --l1-align-bytes flag required by the
	// blend server (blend_server.sh:31).
	cbL1AlignBytes = "16777216"

	// cbDefaultImageRepo is the default engine image repository. CacheBlend runs
	// the same lmcache server binary as LMCacheEngine, so it shares the default
	// image; a private blend image is set via spec.image.
	cbDefaultImageRepo = "lmcache/vllm-openai"

	// cbKVConnector is the connector name injected into vLLM pods bound to a
	// CacheBlendEngine (design §7).
	cbKVConnector = "CBKVConnector"

	// cbKVConnectorModulePath is the import path of the CacheBlend connector
	// (cacheblend-plugin/README.md:66).
	cbKVConnectorModulePath = "lmcache_cacheblend.connector"
)

// cbSpecToEngineSpec projects a CacheBlendEngineSpec onto an LMCacheEngineSpec so
// the shared, spec-keyed resource builders (ComputeResources, BuildContainerArgs,
// and the buildDaemonSetCore/buildLookupServiceCore/buildMetricsServiceCore
// scaffolding) can be reused without duplicating the GPU/security pod template.
//
// CacheBlendEngineSpec deliberately reuses the same shared sub-structs as
// LMCacheEngineSpec (ServerSpec, L1BackendSpec, EvictionSpec, PrometheusSpec,
// L2BackendSpec, ImageSpec, etc.), so every field consumed by the shared builders
// maps across one-to-one. The CacheBlend-only fields (Blend, Injection) are not
// consumed by the engine builders and are intentionally dropped here; they are
// surfaced separately (Blend via BuildCBConnectionConfigMap, Injection via the
// admission webhook).
func cbSpecToEngineSpec(spec *lmcachev1alpha1.CacheBlendEngineSpec) *lmcachev1alpha1.LMCacheEngineSpec {
	return &lmcachev1alpha1.LMCacheEngineSpec{
		GPUVendor:          spec.GPUVendor,
		Image:              spec.Image,
		ImagePullSecrets:   spec.ImagePullSecrets,
		Server:             spec.Server,
		L1:                 spec.L1,
		Eviction:           spec.Eviction,
		Prometheus:         spec.Prometheus,
		L2Backend:          spec.L2Backend,
		Coordinator:        spec.Coordinator,
		ResourceOverrides:  spec.ResourceOverrides,
		LogLevel:           spec.LogLevel,
		NodeSelector:       spec.NodeSelector,
		Affinity:           spec.Affinity,
		Tolerations:        spec.Tolerations,
		Env:                spec.Env,
		Volumes:            spec.Volumes,
		VolumeMounts:       spec.VolumeMounts,
		PodAnnotations:     spec.PodAnnotations,
		PodLabels:          spec.PodLabels,
		ServiceAccountName: spec.ServiceAccountName,
		PriorityClassName:  spec.PriorityClassName,
		Privileged:         spec.Privileged,
		ExtraArgs:          spec.ExtraArgs,
	}
}

// BuildCBEngineArgs returns the server CLI args for the blend_v3 engine: the
// proven LMCacheEngine serialization (--host/--port/--l1-size-gb/--chunk-size/
// eviction/prometheus/L2) plus the CacheBlend-specific --engine-type blend and
// --l1-align-bytes flags. The blend flags are inserted before the user-supplied
// extraArgs so a user can still override them.
func BuildCBEngineArgs(spec *lmcachev1alpha1.CacheBlendEngineSpec) []string {
	// Serialize the base server args WITHOUT the user extraArgs, append the blend
	// flags, then append extraArgs last so they retain their override precedence.
	base := cbSpecToEngineSpec(spec)
	base.ExtraArgs = nil
	args := BuildContainerArgs(base)
	args = append(args,
		"--engine-type", cbEngineType,
		"--l1-align-bytes", cbL1AlignBytes,
	)
	args = append(args, spec.ExtraArgs...)
	return args
}

// BuildCBEngineDaemonSet constructs the DaemonSet for the blend_v3 engine of the
// given CacheBlendEngine. It reuses the shared GPU/security pod-template
// scaffolding (hostIPC, runtimeClassName=nvidia, NVIDIA_VISIBLE_DEVICES=all,
// privileged only when spec.privileged=true (default false), CPU+memory-only
// resources with no nvidia.com/gpu claim) so the engine shares the node's GPU
// via CUDA IPC, and adds the blend-specific server args.
func BuildCBEngineDaemonSet(engine *lmcachev1alpha1.CacheBlendEngine) *appsv1.DaemonSet {
	engineSpec := cbSpecToEngineSpec(&engine.Spec)
	return buildDaemonSetCore(
		engine.Name,
		engine.Namespace,
		engineSpec,
		BuildCBEngineArgs(&engine.Spec),
		cbDefaultImageRepo,
	)
}

// BuildCBEngineLookupService creates the node-local lookup Service
// (internalTrafficPolicy=Local) for the CacheBlendEngine, so opted-in vLLM pods
// reach the blend_v3 engine on their own node.
func BuildCBEngineLookupService(engine *lmcachev1alpha1.CacheBlendEngine) *corev1.Service {
	return buildLookupServiceCore(engine.Name, engine.Namespace, cbSpecToEngineSpec(&engine.Spec))
}

// BuildCBEngineMetricsService creates the headless metrics Service for the
// CacheBlendEngine.
func BuildCBEngineMetricsService(engine *lmcachev1alpha1.CacheBlendEngine) *corev1.Service {
	return buildMetricsServiceCore(engine.Name, engine.Namespace, cbSpecToEngineSpec(&engine.Spec))
}

// BuildCBConnectionConfigMap creates the <engine>-connection ConfigMap carrying
// the CBKVConnector kv-transfer-config JSON (design §7). The JSON points vLLM at
// the node-local Service (lmcache.mp.host) and carries the blend tunables
// cb.check_layer and cb.recomp_ratio read from spec.Blend (defaults are pinned by
// SetDefaults: checkLayer=1, recompRatio=0.15).
func BuildCBConnectionConfigMap(engine *lmcachev1alpha1.CacheBlendEngine) *corev1.ConfigMap {
	spec := &engine.Spec
	// Use the same default (5555) as BuildContainerArgs/getServerPort so the
	// connection ConfigMap's lmcache.mp.port always matches the engine's actual
	// --port; the two artifacts must never drift (design §9.10).
	port := derefInt32(getServerPort(cbSpecToEngineSpec(spec)), 5555)

	checkLayer := int32(1)
	recompRatio := 0.15
	if spec.Blend != nil {
		checkLayer = derefInt32(spec.Blend.CheckLayer, 1)
		recompRatio = derefFloat64(spec.Blend.RecompRatio, 0.15)
	}

	extra := map[string]any{
		"cb.check_layer":  checkLayer,
		"cb.recomp_ratio": recompRatio,
	}

	return buildConnectionConfigMapCore(
		engine.Name,
		engine.Namespace,
		cbKVConnector,
		cbKVConnectorModulePath,
		port,
		extra,
	)
}

// CBServiceMonitorEnabled reports whether a ServiceMonitor should be created for
// the given CacheBlendEngine. It reuses the shared spec-keyed predicate since
// CacheBlendEngineSpec embeds the same PrometheusSpec sub-struct.
func CBServiceMonitorEnabled(engine *lmcachev1alpha1.CacheBlendEngine) bool {
	return ServiceMonitorEnabled(cbSpecToEngineSpec(&engine.Spec))
}

// BuildCBServiceMonitor creates the ServiceMonitor CR for the CacheBlendEngine,
// reusing the shared name/namespace/spec-keyed core. Callers must ensure
// CBServiceMonitorEnabled(engine) is true.
func BuildCBServiceMonitor(engine *lmcachev1alpha1.CacheBlendEngine) *monitoringv1.ServiceMonitor {
	return buildServiceMonitorCore(engine.Name, engine.Namespace, cbSpecToEngineSpec(&engine.Spec))
}
