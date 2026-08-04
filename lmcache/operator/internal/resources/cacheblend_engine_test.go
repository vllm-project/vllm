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
	"slices"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// minimalCBEngine returns a CacheBlendEngine with only the required L1 size set,
// then applies SetDefaults so blend/injection/chunk-size defaults are pinned —
// matching how the controller hands the object to the builders.
func minimalCBEngine() *lmcachev1alpha1.CacheBlendEngine {
	e := &lmcachev1alpha1.CacheBlendEngine{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testEngineName,
			Namespace: testNamespace,
		},
		Spec: lmcachev1alpha1.CacheBlendEngineSpec{
			L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		},
	}
	e.SetDefaults()
	return e
}

// ===========================
// BuildCBEngineArgs
// ===========================

func TestBuildCBEngineArgs_BlendFlags(t *testing.T) {
	args := BuildCBEngineArgs(&minimalCBEngine().Spec)

	// Blend-specific flags.
	assertArg(t, args, "--engine-type", "blend")
	assertArg(t, args, "--l1-align-bytes", "16777216")

	// Reuses the proven LMCacheEngine serialization (NOT --l1-size).
	assertArg(t, args, "--l1-size-gb", "10.0")
	assertNoArg(t, args, "--l1-size")

	// chunk-size pinned to 256 by SetDefaults.
	assertArg(t, args, "--chunk-size", "256")

	// Standard server args carried over.
	assertArg(t, args, "--host", "0.0.0.0")
	assertArg(t, args, "--port", "5555")
	assertArg(t, args, "--hash-algorithm", "blake3")
}

func TestBuildCBEngineArgs_ExtraArgsAfterBlendFlags(t *testing.T) {
	engine := minimalCBEngine()
	engine.Spec.ExtraArgs = []string{"--engine-type", "override-me"}

	args := BuildCBEngineArgs(&engine.Spec)

	// The user's extraArgs are appended last so they can override the blend
	// defaults; assert the override value appears after the operator-set one.
	firstIdx := slices.Index(args, "--engine-type")
	lastIdx := -1
	for i, a := range args {
		if a == "--engine-type" {
			lastIdx = i
		}
	}
	if firstIdx == lastIdx {
		t.Fatalf("expected --engine-type to appear twice, got args=%v", args)
	}
	if args[firstIdx+1] != "blend" {
		t.Fatalf("expected operator-set --engine-type blend first, got %s", args[firstIdx+1])
	}
	if args[lastIdx+1] != "override-me" {
		t.Fatalf("expected user --engine-type override-me last, got %s", args[lastIdx+1])
	}
}

// ===========================
// BuildCBEngineDaemonSet
// ===========================

func TestBuildCBEngineDaemonSet_GPUAndSecurity(t *testing.T) {
	engine := minimalCBEngine()
	ds := BuildCBEngineDaemonSet(engine)

	if ds.Name != testEngineName {
		t.Fatalf("expected name %s, got %s", testEngineName, ds.Name)
	}
	if ds.Namespace != testNamespace {
		t.Fatalf("expected namespace %s, got %s", testNamespace, ds.Namespace)
	}

	podSpec := ds.Spec.Template.Spec

	// hostIPC required for CUDA IPC with the node-local engine.
	if !podSpec.HostIPC {
		t.Fatal("expected HostIPC=true")
	}
	// runtimeClassName=nvidia for the default (nvidia) vendor.
	if podSpec.RuntimeClassName == nil || *podSpec.RuntimeClassName != nvidiaRuntimeClass {
		t.Fatalf("expected RuntimeClassName=nvidia, got %v", podSpec.RuntimeClassName)
	}

	if len(podSpec.Containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(podSpec.Containers))
	}
	c := podSpec.Containers[0]

	// privileged defaults to false (opt-in via spec.privileged).
	if c.SecurityContext == nil || c.SecurityContext.Privileged == nil || *c.SecurityContext.Privileged {
		t.Fatal("expected privileged=false by default")
	}

	// NVIDIA env exposes all GPUs without a device-plugin claim.
	if !hasEnvAll(c.Env, "NVIDIA_VISIBLE_DEVICES") {
		t.Fatal("missing NVIDIA_VISIBLE_DEVICES=all")
	}
	if !hasEnvAll(c.Env, "NVIDIA_DRIVER_CAPABILITIES") {
		t.Fatal("missing NVIDIA_DRIVER_CAPABILITIES=all")
	}

	// Command is the same lmcache server binary as LMCacheEngine.
	if len(c.Command) < 2 || c.Command[0] != lmcacheServerBinary || c.Command[1] != serverSubcommand {
		t.Fatalf("expected lmcache server command, got %v", c.Command)
	}

	// Blend args present on the container.
	assertArg(t, c.Args, "--engine-type", "blend")
	assertArg(t, c.Args, "--l1-align-bytes", "16777216")
}

func TestBuildCBEngineDaemonSet_PrivilegedEnabled(t *testing.T) {
	engine := minimalCBEngine()
	engine.Spec.Privileged = ptr(true)
	ds := BuildCBEngineDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// spec.privileged=true threads through to the container security context.
	if c.SecurityContext == nil || c.SecurityContext.Privileged == nil || !*c.SecurityContext.Privileged {
		t.Fatal("expected privileged=true when spec.privileged=true")
	}
}

func TestBuildCBEngineDaemonSet_NoGPUResourceClaim(t *testing.T) {
	engine := minimalCBEngine()
	ds := BuildCBEngineDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// The engine shares vLLM's GPU via CUDA IPC; it must NOT claim a
	// device-plugin GPU (nvidia.com/gpu) in requests or limits.
	const gpuResource = corev1.ResourceName("nvidia.com/gpu")
	if _, ok := c.Resources.Requests[gpuResource]; ok {
		t.Fatal("nvidia.com/gpu must not be in resource requests")
	}
	if _, ok := c.Resources.Limits[gpuResource]; ok {
		t.Fatal("nvidia.com/gpu must not be in resource limits")
	}

	// CPU + memory ARE present (auto-computed from L1 size = 10).
	if c.Resources.Requests.Cpu().IsZero() {
		t.Fatal("expected a CPU request")
	}
	if c.Resources.Requests.Memory().IsZero() {
		t.Fatal("expected a memory request")
	}
}

func TestBuildCBEngineDaemonSet_ImagePullSecrets(t *testing.T) {
	engine := minimalCBEngine()
	engine.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: "private-engine-reg"}}

	ds := BuildCBEngineDaemonSet(engine)

	secrets := ds.Spec.Template.Spec.ImagePullSecrets
	if len(secrets) != 1 || secrets[0].Name != "private-engine-reg" {
		t.Fatalf("expected engine imagePullSecrets wired onto the pod, got %v", secrets)
	}
}

func TestBuildCBEngineDaemonSet_DefaultImage(t *testing.T) {
	engine := minimalCBEngine()
	ds := BuildCBEngineDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	if c.Image != "lmcache/vllm-openai:latest" {
		t.Fatalf("expected default image lmcache/vllm-openai:latest, got %s", c.Image)
	}
}

func TestBuildCBEngineDaemonSet_AMDNoRuntimeClass(t *testing.T) {
	engine := minimalCBEngine()
	engine.Spec.GPUVendor = ptr(lmcachev1alpha1.GPUVendorAMD)

	ds := BuildCBEngineDaemonSet(engine)
	podSpec := ds.Spec.Template.Spec

	if podSpec.RuntimeClassName != nil {
		t.Fatalf("expected nil RuntimeClassName for AMD, got %q", *podSpec.RuntimeClassName)
	}
	// hostIPC is still injected for AMD (privileged stays at its default false
	// here since the fixture does not opt in).
	if !podSpec.HostIPC {
		t.Fatal("expected HostIPC=true even for AMD")
	}
}

// ===========================
// BuildCBEngineLookupService / MetricsService
// ===========================

func TestBuildCBEngineLookupService_Local(t *testing.T) {
	engine := minimalCBEngine()
	svc := BuildCBEngineLookupService(engine)

	if svc.Name != testEngineName {
		t.Fatalf("expected name %s, got %s", testEngineName, svc.Name)
	}
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		t.Fatal("lookup service must not be headless")
	}
	if svc.Spec.InternalTrafficPolicy == nil ||
		*svc.Spec.InternalTrafficPolicy != corev1.ServiceInternalTrafficPolicyLocal {
		t.Fatal("expected internalTrafficPolicy=Local")
	}
	if svc.Spec.Ports[0].Port != 5555 {
		t.Fatalf("expected server port 5555, got %d", svc.Spec.Ports[0].Port)
	}
}

func TestBuildCBEngineMetricsService_Headless(t *testing.T) {
	engine := minimalCBEngine()
	svc := BuildCBEngineMetricsService(engine)

	if svc.Name != testEngineName+"-metrics" {
		t.Fatalf("expected name %s-metrics, got %s", testEngineName, svc.Name)
	}
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Fatal("expected headless metrics service (ClusterIP=None)")
	}
	if svc.Spec.Ports[0].Port != 9090 {
		t.Fatalf("expected metrics port 9090, got %d", svc.Spec.Ports[0].Port)
	}
}

// ===========================
// BuildCBConnectionConfigMap
// ===========================

// parseCBConnectionConfig unmarshals the CBKVConnector kv-transfer-config JSON
// from the connection ConfigMap and returns the top-level config plus the
// kv_connector_extra_config submap.
func parseCBConnectionConfig(t *testing.T, cm *corev1.ConfigMap) (map[string]any, map[string]any) {
	t.Helper()
	jsonStr, ok := cm.Data["kv-transfer-config.json"]
	if !ok {
		t.Fatal("missing kv-transfer-config.json key")
	}
	var config map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &config); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	extra, ok := config["kv_connector_extra_config"].(map[string]any)
	if !ok {
		t.Fatal("missing kv_connector_extra_config map")
	}
	return config, extra
}

func TestBuildCBConnectionConfigMap_Default(t *testing.T) {
	engine := minimalCBEngine()
	cm := BuildCBConnectionConfigMap(engine)

	if cm.Name != "test-engine-connection" {
		t.Fatalf("expected name test-engine-connection, got %s", cm.Name)
	}
	if cm.Namespace != testNamespace {
		t.Fatalf("expected namespace %s, got %s", testNamespace, cm.Namespace)
	}

	config, extra := parseCBConnectionConfig(t, cm)

	if config["kv_connector"] != "CBKVConnector" {
		t.Fatalf("expected kv_connector=CBKVConnector, got %v", config["kv_connector"])
	}
	if config["kv_connector_module_path"] != "lmcache_cacheblend.connector" {
		t.Fatalf("expected kv_connector_module_path=lmcache_cacheblend.connector, got %v",
			config["kv_connector_module_path"])
	}
	if config["kv_role"] != "kv_both" {
		t.Fatalf("expected kv_role=kv_both, got %v", config["kv_role"])
	}

	if extra["lmcache.mp.host"] != "tcp://test-engine.default.svc.cluster.local" {
		t.Fatalf("expected tcp:// node-local Service host, got %v", extra["lmcache.mp.host"])
	}
	if extra["lmcache.mp.port"] != "5555" {
		t.Fatalf("expected lmcache.mp.port=5555, got %v", extra["lmcache.mp.port"])
	}

	// Blend tunables from SetDefaults (checkLayer=1, recompRatio=0.15). JSON
	// numbers decode to float64.
	if extra["cb.check_layer"] != float64(1) {
		t.Fatalf("expected cb.check_layer=1, got %v", extra["cb.check_layer"])
	}
	if extra["cb.recomp_ratio"] != 0.15 {
		t.Fatalf("expected cb.recomp_ratio=0.15, got %v", extra["cb.recomp_ratio"])
	}
}

func TestBuildCBConnectionConfigMap_CustomBlendAndPort(t *testing.T) {
	engine := minimalCBEngine()
	engine.Spec.Server.Port = ptr(int32(6566))
	engine.Spec.Blend = &lmcachev1alpha1.BlendSpec{
		CheckLayer:  ptr(int32(3)),
		RecompRatio: ptr(0.5),
	}

	cm := BuildCBConnectionConfigMap(engine)
	_, extra := parseCBConnectionConfig(t, cm)

	if extra["lmcache.mp.port"] != "6566" {
		t.Fatalf("expected lmcache.mp.port=6566, got %v", extra["lmcache.mp.port"])
	}
	if extra["cb.check_layer"] != float64(3) {
		t.Fatalf("expected cb.check_layer=3, got %v", extra["cb.check_layer"])
	}
	if extra["cb.recomp_ratio"] != 0.5 {
		t.Fatalf("expected cb.recomp_ratio=0.5, got %v", extra["cb.recomp_ratio"])
	}
}

// TestBuildCBConnectionConfigMap_PortMatchesEngineArgs asserts the connection
// ConfigMap's lmcache.mp.port and the engine DaemonSet's --port never drift, for
// both the default and a user-set port.
func TestBuildCBConnectionConfigMap_PortMatchesEngineArgs(t *testing.T) {
	tests := []struct {
		name     string
		port     *int32
		wantPort string
	}{
		{name: "default", port: nil, wantPort: "5555"},
		{name: "custom", port: ptr(int32(6566)), wantPort: "6566"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := minimalCBEngine()
			engine.Spec.Server.Port = tt.port

			args := BuildCBEngineArgs(&engine.Spec)
			cm := BuildCBConnectionConfigMap(engine)
			_, extra := parseCBConnectionConfig(t, cm)

			assertArg(t, args, "--port", tt.wantPort)
			if extra["lmcache.mp.port"] != tt.wantPort {
				t.Fatalf("connection port %v != engine --port %s", extra["lmcache.mp.port"], tt.wantPort)
			}
		})
	}
}

// TestBuildCBEngine_ChunkSizeConsistency asserts the engine's chunk-size is 256
// (the only value CacheBlend supports — block_size 64 * 4), matching the locked
// CacheBlendChunkSize constant, so it cannot drift from the injected --block-size.
func TestBuildCBEngine_ChunkSizeConsistency(t *testing.T) {
	engine := minimalCBEngine()
	args := BuildCBEngineArgs(&engine.Spec)

	assertArg(t, args, "--chunk-size", "256")
	if lmcachev1alpha1.CacheBlendChunkSize != 256 {
		t.Fatalf("CacheBlendChunkSize constant must be 256, got %d", lmcachev1alpha1.CacheBlendChunkSize)
	}
}
