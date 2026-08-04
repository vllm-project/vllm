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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	testEngineName = "test-engine"
	testNamespace  = "default"
)

// --- helpers ---

func ptr[T any](v T) *T { return &v }

func findArgValue(t *testing.T, args []string, flag string) string { //nolint:unparam // test helper, flag varies by caller
	t.Helper()
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	t.Fatalf("expected %s flag in args", flag)
	return ""
}

func minimalEngine() *lmcachev1alpha1.LMCacheEngine {
	return &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testEngineName,
			Namespace: testNamespace,
		},
		Spec: lmcachev1alpha1.LMCacheEngineSpec{
			L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		},
	}
}

// ===========================
// Deref helpers
// ===========================

func TestDerefInt32(t *testing.T) {
	if v := derefInt32(ptr(int32(42)), 0); v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
	if v := derefInt32(nil, 99); v != 99 {
		t.Fatalf("expected 99, got %d", v)
	}
}

func TestDerefString(t *testing.T) {
	if v := derefString(ptr("hello"), "x"); v != "hello" {
		t.Fatalf("expected hello, got %s", v)
	}
	if v := derefString(nil, testNamespace); v != testNamespace {
		t.Fatalf("expected default, got %s", v)
	}
}

func TestDerefBool(t *testing.T) {
	if v := derefBool(ptr(true), false); !v {
		t.Fatal("expected true")
	}
	if v := derefBool(nil, true); !v {
		t.Fatal("expected true (default)")
	}
	if v := derefBool(nil, false); v {
		t.Fatal("expected false (default)")
	}
}

func TestDerefFloat64(t *testing.T) {
	if v := derefFloat64(ptr(3.14), 0); v != 3.14 {
		t.Fatalf("expected 3.14, got %f", v)
	}
	if v := derefFloat64(nil, 2.71); v != 2.71 {
		t.Fatalf("expected 2.71, got %f", v)
	}
}

// ===========================
// Labels
// ===========================

func TestSelectorLabels(t *testing.T) {
	labels := SelectorLabels("my-cache")
	expected := map[string]string{
		"app.kubernetes.io/name":       "lmcache",
		"app.kubernetes.io/instance":   "my-cache",
		"app.kubernetes.io/managed-by": "lmcache-operator",
	}
	if len(labels) != len(expected) {
		t.Fatalf("expected %d labels, got %d", len(expected), len(labels))
	}
	for k, v := range expected {
		if labels[k] != v {
			t.Errorf("label %s: expected %s, got %s", k, v, labels[k])
		}
	}
}

func TestStandardLabels(t *testing.T) {
	labels := StandardLabels("my-cache")
	if labels["app.kubernetes.io/component"] != "cache-engine" {
		t.Fatal("missing component label")
	}
	// Should include selector labels too
	if labels["app.kubernetes.io/instance"] != "my-cache" {
		t.Fatal("missing instance label")
	}
}

func TestMergeLabels(t *testing.T) {
	a := map[string]string{"k1": "v1", "k2": "v2"}
	b := map[string]string{"k2": "override", "k3": "v3"}
	merged := MergeLabels(a, b)

	if merged["k1"] != "v1" {
		t.Error("k1 should be v1")
	}
	if merged["k2"] != "override" {
		t.Error("k2 should be overridden to 'override'")
	}
	if merged["k3"] != "v3" {
		t.Error("k3 should be v3")
	}
}

func TestMergeLabels_Empty(t *testing.T) {
	merged := MergeLabels()
	if len(merged) != 0 {
		t.Fatal("expected empty map")
	}
}

func TestMergeLabels_NilMaps(t *testing.T) {
	merged := MergeLabels(nil, map[string]string{"k": "v"}, nil)
	if merged["k"] != "v" {
		t.Fatal("expected k=v")
	}
}

// ===========================
// ComputeResources
// ===========================

func TestComputeResources_AutoComputed(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
	}
	res := ComputeResources(spec)

	// memReq = ceil(10+5) = 15Gi
	expectedReq := resource.MustParse("15Gi")
	if !res.Requests.Memory().Equal(expectedReq) {
		t.Fatalf("expected memory request 15Gi, got %s", res.Requests.Memory())
	}

	// memLim = ceil(15*1.5) = 23Gi
	expectedLim := resource.MustParse("23Gi")
	if !res.Limits.Memory().Equal(expectedLim) {
		t.Fatalf("expected memory limit 23Gi, got %s", res.Limits.Memory())
	}

	// CPU request = 4
	expectedCPU := resource.MustParse("4")
	if !res.Requests.Cpu().Equal(expectedCPU) {
		t.Fatalf("expected cpu request 4, got %s", res.Requests.Cpu())
	}
}

func TestComputeResources_FractionalSizeGB(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 0.5},
	}
	res := ComputeResources(spec)

	// memReq = ceil(0.5+5) = 6Gi
	expectedReq := resource.MustParse("6Gi")
	if !res.Requests.Memory().Equal(expectedReq) {
		t.Fatalf("expected memory request 6Gi, got %s", res.Requests.Memory())
	}

	// memLim = ceil(6*1.5) = 9Gi
	expectedLim := resource.MustParse("9Gi")
	if !res.Limits.Memory().Equal(expectedLim) {
		t.Fatalf("expected memory limit 9Gi, got %s", res.Limits.Memory())
	}
}

func TestComputeResources_Override(t *testing.T) {
	override := &corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		},
	}
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1:                lmcachev1alpha1.L1BackendSpec{SizeGB: 100},
		ResourceOverrides: override,
	}
	res := ComputeResources(spec)
	if !res.Requests.Memory().Equal(resource.MustParse("1Gi")) {
		t.Fatalf("expected override to apply, got %s", res.Requests.Memory())
	}
}

// ===========================
// BuildContainerArgs
// ===========================

func TestBuildContainerArgs_Defaults(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
	}
	args := BuildContainerArgs(spec)

	assertArg(t, args, "--host", "0.0.0.0")
	assertArg(t, args, "--port", "5555")
	assertArg(t, args, "--l1-size-gb", "10.0")
	assertArg(t, args, "--chunk-size", "256")
	assertArg(t, args, "--max-workers", "1")
	assertArg(t, args, "--hash-algorithm", "blake3")
	assertArg(t, args, "--eviction-policy", "LRU")
	assertArg(t, args, "--eviction-trigger-watermark", "0.80")
	assertArg(t, args, "--eviction-ratio", "0.20")
	assertArg(t, args, "--prometheus-port", "9090")
	assertNoArg(t, args, "--disable-prometheus")
}

func TestBuildContainerArgs_CustomServer(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 5},
		Server: &lmcachev1alpha1.ServerSpec{
			Port:          ptr(int32(8080)),
			ChunkSize:     ptr(int32(512)),
			MaxWorkers:    ptr(int32(4)),
			HashAlgorithm: ptr("sha256_cbor"),
		},
	}
	args := BuildContainerArgs(spec)

	assertArg(t, args, "--port", "8080")
	assertArg(t, args, "--chunk-size", "512")
	assertArg(t, args, "--max-workers", "4")
	assertArg(t, args, "--hash-algorithm", "sha256_cbor")
}

func TestBuildContainerArgs_CustomEviction(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		Eviction: &lmcachev1alpha1.EvictionSpec{
			Policy:           ptr("LRU"),
			TriggerWatermark: ptr(0.9),
			EvictionRatio:    ptr(0.3),
		},
	}
	args := BuildContainerArgs(spec)

	assertArg(t, args, "--eviction-policy", "LRU")
	assertArg(t, args, "--eviction-trigger-watermark", "0.90")
	assertArg(t, args, "--eviction-ratio", "0.30")
}

func TestBuildContainerArgs_PrometheusDisabled(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		Prometheus: &lmcachev1alpha1.PrometheusSpec{
			Enabled: ptr(false),
		},
	}
	args := BuildContainerArgs(spec)

	assertHasArg(t, args, "--disable-prometheus")
	assertNoArg(t, args, "--prometheus-port")
}

func TestBuildContainerArgs_CustomPrometheusPort(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		Prometheus: &lmcachev1alpha1.PrometheusSpec{
			Enabled: ptr(true),
			Port:    ptr(int32(8888)),
		},
	}
	args := BuildContainerArgs(spec)

	assertArg(t, args, "--prometheus-port", "8888")
}

func TestBuildContainerArgs_L2RESP(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis.default.svc",
				Port: 6379,
			},
		},
	}
	args := BuildContainerArgs(spec)

	// Default policies should be present
	assertArg(t, args, "--l2-store-policy", "default")
	assertArg(t, args, "--l2-prefetch-policy", "default")
	assertArg(t, args, "--l2-prefetch-max-in-flight", "8")

	l2JSON := findArgValue(t, args, "--l2-adapter")

	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if parsed["type"] != "resp" {
		t.Fatalf("expected type=resp, got %v", parsed["type"])
	}
	if parsed["host"] != "redis.default.svc" {
		t.Fatalf("expected host=redis.default.svc, got %v", parsed["host"])
	}
	if parsed["port"] != float64(6379) {
		t.Fatalf("expected port=6379, got %v", parsed["port"])
	}
	if parsed["num_workers"] != float64(8) {
		t.Fatalf("expected num_workers=8, got %v", parsed["num_workers"])
	}
	// No auth — username/password should be absent
	if _, ok := parsed["username"]; ok {
		t.Fatal("expected no username without authSecretRef")
	}
}

func TestBuildContainerArgs_L2RESPWithAuth(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis",
				Port: 6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{
					Name: "redis-auth",
				},
			},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")

	// Credentials should NOT be in the JSON — they are passed via env vars.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if _, ok := parsed["username"]; ok {
		t.Fatal("username should not be in L2 JSON — it is passed via env var")
	}
	if _, ok := parsed["password"]; ok {
		t.Fatal("password should not be in L2 JSON — it is passed via env var")
	}
}

func TestBuildContainerArgs_L2RESPCustomWorkers(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host:          "redis",
				Port:          6379,
				NumWorkers:    ptr(int32(16)),
				MaxCapacityGB: ptr(50.0),
			},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")

	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if parsed["num_workers"] != float64(16) {
		t.Fatalf("expected num_workers=16, got %v", parsed["num_workers"])
	}
	if parsed["max_capacity_gb"] != float64(50) {
		t.Fatalf("expected max_capacity_gb=50, got %v", parsed["max_capacity_gb"])
	}
}

func TestBuildContainerArgs_L2Raw(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			Raw: &lmcachev1alpha1.RawL2AdapterSpec{
				Type: "mock",
				Config: map[string]apiextensionsv1.JSON{
					"max_size_gb":       {Raw: []byte(`256`)},
					"mock_bandwidth_gb": {Raw: []byte(`10`)},
				},
			},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")

	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if parsed["type"] != "mock" {
		t.Fatalf("expected type=mock, got %v", parsed["type"])
	}
	if parsed["max_size_gb"] != float64(256) {
		t.Fatalf("expected max_size_gb=256, got %v", parsed["max_size_gb"])
	}
}

func TestBuildContainerArgs_L2SerdeAESGCMRESP(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{Host: "redis", Port: 6379},
			Serde: &lmcachev1alpha1.L2SerdeSpec{
				AESGCM: &lmcachev1alpha1.AESGCMSerdeSpec{
					MasterKeySecretRef: corev1.LocalObjectReference{Name: "l2-master-key"},
				},
			},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")
	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	serde, ok := parsed["serde"].(map[string]any)
	if !ok {
		t.Fatalf("expected serde sub-dict, got %v", parsed["serde"])
	}
	if serde["type"] != "aesgcm" {
		t.Fatalf("expected serde type=aesgcm, got %v", serde["type"])
	}
	if serde["key_provider"] != "hkdf" {
		t.Fatalf("expected key_provider=hkdf, got %v", serde["key_provider"])
	}
	if serde["master_key_path"] != l2EncryptionKeyPath {
		t.Fatalf("expected master_key_path=%s, got %v", l2EncryptionKeyPath, serde["master_key_path"])
	}
	if serde["aes_bits"] != float64(128) {
		t.Fatalf("expected aes_bits=128, got %v", serde["aes_bits"])
	}
}

func TestBuildContainerArgs_L2SerdeAESGCMRawWithOverrides(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			Raw: &lmcachev1alpha1.RawL2AdapterSpec{
				Type: "fs",
				Config: map[string]apiextensionsv1.JSON{
					"base_path": {Raw: []byte(`"/data/l2"`)},
				},
			},
			Serde: &lmcachev1alpha1.L2SerdeSpec{
				AESGCM: &lmcachev1alpha1.AESGCMSerdeSpec{
					MasterKeySecretRef: corev1.LocalObjectReference{Name: "l2-master-key"},
					AESBits:            ptr(int32(256)),
				},
			},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")
	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if parsed["base_path"] != "/data/l2" {
		t.Fatalf("expected base_path=/data/l2, got %v", parsed["base_path"])
	}
	serde, ok := parsed["serde"].(map[string]any)
	if !ok {
		t.Fatalf("expected serde sub-dict, got %v", parsed["serde"])
	}
	if serde["aes_bits"] != float64(256) {
		t.Fatalf("expected aes_bits=256, got %v", serde["aes_bits"])
	}
}

func TestBuildContainerArgs_L2NoSerdeConfigured(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{Host: "redis", Port: 6379},
		},
	}
	args := BuildContainerArgs(spec)

	l2JSON := findArgValue(t, args, "--l2-adapter")
	var parsed map[string]any
	if err := json.Unmarshal([]byte(l2JSON), &parsed); err != nil {
		t.Fatalf("failed to parse L2 JSON: %v", err)
	}
	if _, ok := parsed["serde"]; ok {
		t.Fatal("expected no serde sub-dict when none is configured")
	}
}

func TestBuildContainerArgs_L2CustomPolicies(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		L2Backend: &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis",
				Port: 6379,
			},
			StorePolicy:         ptr("skip_l1"),
			PrefetchPolicy:      ptr("default"),
			PrefetchMaxInFlight: ptr(int32(16)),
		},
	}
	args := BuildContainerArgs(spec)

	assertArg(t, args, "--l2-store-policy", "skip_l1")
	assertArg(t, args, "--l2-prefetch-policy", "default")
	assertArg(t, args, "--l2-prefetch-max-in-flight", "16")
}

func TestBuildContainerArgs_NoL2Backend(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
	}
	args := BuildContainerArgs(spec)

	for _, a := range args {
		if a == "--l2-adapter" {
			t.Fatal("expected no --l2-adapter flag when L2Backend is nil")
		}
		if a == "--l2-store-policy" {
			t.Fatal("expected no --l2-store-policy flag when L2Backend is nil")
		}
	}
}

func TestBuildContainerArgs_ExtraArgs(t *testing.T) {
	spec := &lmcachev1alpha1.LMCacheEngineSpec{
		L1:        lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
		ExtraArgs: []string{"--l1-init-size-gb", "5", "--custom-flag", "value"},
	}
	args := BuildContainerArgs(spec)

	// extraArgs should be appended at the end
	assertArg(t, args, "--l1-init-size-gb", "5")
	assertArg(t, args, "--custom-flag", "value")
}

// ===========================
// BuildDaemonSet
// ===========================

func TestBuildDaemonSet_Minimal(t *testing.T) {
	engine := minimalEngine()
	ds := BuildDaemonSet(engine)

	if ds.Name != testEngineName {
		t.Fatalf("expected name test-engine, got %s", ds.Name)
	}
	if ds.Namespace != testNamespace {
		t.Fatalf("expected namespace default, got %s", ds.Namespace)
	}

	// Should have selector labels
	sel := ds.Spec.Selector.MatchLabels
	if sel["app.kubernetes.io/instance"] != testEngineName {
		t.Fatal("missing instance selector label")
	}

	// Should NOT have HostNetwork (uses Service with internalTrafficPolicy=Local instead)
	if ds.Spec.Template.Spec.HostNetwork {
		t.Fatal("expected HostNetwork=false")
	}

	// Should have HostIPC=true (required for CUDA IPC)
	if !ds.Spec.Template.Spec.HostIPC {
		t.Fatal("expected HostIPC=true")
	}

	// Should have exactly 1 container
	if len(ds.Spec.Template.Spec.Containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(ds.Spec.Template.Spec.Containers))
	}

	c := ds.Spec.Template.Spec.Containers[0]
	if c.Name != "lmcache" {
		t.Fatalf("expected container name lmcache, got %s", c.Name)
	}
	if c.Image != "lmcache/vllm-openai:latest" {
		t.Fatalf("expected default image, got %s", c.Image)
	}
	if c.ImagePullPolicy != corev1.PullIfNotPresent {
		t.Fatalf("expected PullIfNotPresent, got %s", c.ImagePullPolicy)
	}

	// Should have probes
	if c.StartupProbe == nil {
		t.Fatal("missing startup probe")
	}
	if c.LivenessProbe == nil {
		t.Fatal("missing liveness probe")
	}
	if c.ReadinessProbe == nil {
		t.Fatal("missing readiness probe")
	}

	// Should NOT have emptyDir /dev/shm volume (hostIPC provides host's /dev/shm)
	for _, v := range ds.Spec.Template.Spec.Volumes {
		if v.Name == "dshm" {
			t.Fatal("dshm emptyDir volume should not be present — it shadows host /dev/shm and breaks CUDA IPC")
		}
	}

	// Should have no volumes by default (no user-specified volumes)
	if len(ds.Spec.Template.Spec.Volumes) != 0 {
		t.Fatalf("expected 0 volumes, got %d", len(ds.Spec.Template.Spec.Volumes))
	}

	// Should have no volume mounts by default
	if len(c.VolumeMounts) != 0 {
		t.Fatalf("expected 0 volume mounts, got %d", len(c.VolumeMounts))
	}

	// Should have LMCACHE_LOG_LEVEL env var
	foundLogLevel := false
	foundVisibleDevices := false
	for _, e := range c.Env {
		if e.Name == "LMCACHE_LOG_LEVEL" && e.Value == "INFO" {
			foundLogLevel = true
		}
		if e.Name == "NVIDIA_VISIBLE_DEVICES" && e.Value == "all" {
			foundVisibleDevices = true
		}
	}
	if !foundLogLevel {
		t.Fatal("missing LMCACHE_LOG_LEVEL env var")
	}
	if !foundVisibleDevices {
		t.Fatal("missing NVIDIA_VISIBLE_DEVICES env var")
	}

	// Should have 3 ports (server + http + metrics) by default
	if len(c.Ports) != 3 {
		t.Fatalf("expected 3 container ports, got %d", len(c.Ports))
	}
}

func TestBuildDaemonSet_CustomImage(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Image = &lmcachev1alpha1.ImageSpec{
		Repository: ptr("custom/image"),
		Tag:        ptr("v1.0"),
		PullPolicy: ptr("Always"),
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	if c.Image != "custom/image:v1.0" {
		t.Fatalf("expected custom/image:v1.0, got %s", c.Image)
	}
	if c.ImagePullPolicy != corev1.PullAlways {
		t.Fatalf("expected PullAlways, got %s", c.ImagePullPolicy)
	}
}

func TestBuildDaemonSet_NeverPullPolicy(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Image = &lmcachev1alpha1.ImageSpec{
		PullPolicy: ptr("Never"),
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]
	if c.ImagePullPolicy != corev1.PullNever {
		t.Fatalf("expected PullNever, got %s", c.ImagePullPolicy)
	}
}

func TestBuildDaemonSet_PrometheusDisabled(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Prometheus = &lmcachev1alpha1.PrometheusSpec{
		Enabled: ptr(false),
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// Should have server + http ports, no metrics port
	if len(c.Ports) != 2 {
		t.Fatalf("expected 2 container ports, got %d", len(c.Ports))
	}
	if c.Ports[0].Name != serverPortName {
		t.Fatalf("expected port name 'server', got %s", c.Ports[0].Name)
	}
}

func TestBuildDaemonSet_CustomEnvAndVolumes(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Env = []corev1.EnvVar{
		{Name: "CUSTOM_VAR", Value: "custom_value"},
	}
	engine.Spec.Volumes = []corev1.Volume{
		{Name: "extra-vol", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	}
	engine.Spec.VolumeMounts = []corev1.VolumeMount{
		{Name: "extra-vol", MountPath: "/extra"},
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// Should have LMCACHE_LOG_LEVEL + NVIDIA_VISIBLE_DEVICES + NVIDIA_DRIVER_CAPABILITIES + CUSTOM_VAR
	if len(c.Env) != 4 {
		t.Fatalf("expected 4 env vars, got %d", len(c.Env))
	}

	// Should have only user-specified extra-vol (no built-in dshm)
	if len(ds.Spec.Template.Spec.Volumes) != 1 {
		t.Fatalf("expected 1 volume, got %d", len(ds.Spec.Template.Spec.Volumes))
	}

	// Should have only user-specified extra mount
	if len(c.VolumeMounts) != 1 {
		t.Fatalf("expected 1 volume mount, got %d", len(c.VolumeMounts))
	}
}

func TestBuildDaemonSet_NodeSelectorAndTolerations(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.present": "true"}
	engine.Spec.Tolerations = []corev1.Toleration{
		{Key: "gpu", Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule},
	}
	engine.Spec.ServiceAccountName = "my-sa"
	engine.Spec.PriorityClassName = "high-priority"

	ds := BuildDaemonSet(engine)
	podSpec := ds.Spec.Template.Spec

	if podSpec.NodeSelector["nvidia.com/gpu.present"] != "true" {
		t.Fatal("missing node selector")
	}
	if len(podSpec.Tolerations) != 1 {
		t.Fatal("missing tolerations")
	}
	if podSpec.ServiceAccountName != "my-sa" {
		t.Fatalf("expected SA my-sa, got %s", podSpec.ServiceAccountName)
	}
	if podSpec.PriorityClassName != "high-priority" {
		t.Fatalf("expected priority class high-priority, got %s", podSpec.PriorityClassName)
	}
}

func TestBuildDaemonSet_PodLabelsAndAnnotations(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.PodLabels = map[string]string{"custom": "label"}
	engine.Spec.PodAnnotations = map[string]string{"anno": "value"}

	ds := BuildDaemonSet(engine)

	if ds.Spec.Template.Labels["custom"] != "label" {
		t.Fatal("missing custom pod label")
	}
	if ds.Spec.Template.Annotations["anno"] != "value" {
		t.Fatal("missing custom pod annotation")
	}
}

func TestBuildDaemonSet_RESPNoAuth(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
		RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
			Host: "redis",
			Port: 6379,
		},
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// Should use direct command, not shell wrapper
	if c.Command[0] != lmcacheServerBinary || c.Command[1] != serverSubcommand {
		t.Fatalf("expected lmcache server command, got %v", c.Command)
	}

	// Should have --l2-adapter in args
	if !slices.Contains(c.Args, "--l2-adapter") {
		t.Fatal("expected --l2-adapter in args")
	}

	// Should NOT have RESP auth env vars
	for _, e := range c.Env {
		if e.Name == "LMCACHE_RESP_USERNAME" || e.Name == "LMCACHE_RESP_PASSWORD" {
			t.Fatalf("unexpected auth env var %s without authSecretRef", e.Name)
		}
	}
}

func TestBuildDaemonSet_RESPWithAuth(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
		RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
			Host: "redis",
			Port: 6379,
			AuthSecretRef: &lmcachev1alpha1.SecretReference{
				Name: "redis-auth",
			},
		},
	}

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	// Should use direct python command (no shell wrapper needed)
	if c.Command[0] != lmcacheServerBinary || c.Command[1] != serverSubcommand {
		t.Fatalf("expected lmcache server command, got %v", c.Command)
	}

	// Should have RESP auth env vars from secret
	foundUser := false
	foundPass := false
	for _, e := range c.Env {
		if e.Name == "LMCACHE_RESP_USERNAME" {
			foundUser = true
			if e.ValueFrom == nil || e.ValueFrom.SecretKeyRef == nil {
				t.Fatal("expected secretKeyRef for LMCACHE_RESP_USERNAME")
			}
			expectedSecret := RESPAuthSecretName(testEngineName)
			if e.ValueFrom.SecretKeyRef.Name != expectedSecret {
				t.Fatalf("expected secret name %s, got %s", expectedSecret, e.ValueFrom.SecretKeyRef.Name)
			}
			if e.ValueFrom.SecretKeyRef.Key != "username" {
				t.Fatalf("expected key username, got %s", e.ValueFrom.SecretKeyRef.Key)
			}
		}
		if e.Name == "LMCACHE_RESP_PASSWORD" {
			foundPass = true
			if e.ValueFrom == nil || e.ValueFrom.SecretKeyRef == nil {
				t.Fatal("expected secretKeyRef for LMCACHE_RESP_PASSWORD")
			}
			if e.ValueFrom.SecretKeyRef.Key != "password" {
				t.Fatalf("expected key password, got %s", e.ValueFrom.SecretKeyRef.Key)
			}
		}
	}
	if !foundUser {
		t.Fatal("missing LMCACHE_RESP_USERNAME env var")
	}
	if !foundPass {
		t.Fatal("missing LMCACHE_RESP_PASSWORD env var")
	}
}

func TestBuildDaemonSet_L2AESGCMSerdeMountsMasterKey(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
		RESP: &lmcachev1alpha1.RESPL2AdapterSpec{Host: "redis", Port: 6379},
		Serde: &lmcachev1alpha1.L2SerdeSpec{
			AESGCM: &lmcachev1alpha1.AESGCMSerdeSpec{
				MasterKeySecretRef: corev1.LocalObjectReference{Name: "l2-master-key"},
			},
		},
	}

	ds := BuildDaemonSet(engine)
	pod := ds.Spec.Template.Spec

	var vol *corev1.Volume
	for i := range pod.Volumes {
		if pod.Volumes[i].Name == l2EncryptionKeyVolumeName {
			vol = &pod.Volumes[i]
		}
	}
	if vol == nil {
		t.Fatal("expected l2-master-key volume")
	}
	if vol.Secret == nil || vol.Secret.SecretName != "l2-master-key" {
		t.Fatalf("expected secret volume referencing l2-master-key, got %+v", vol.VolumeSource)
	}
	// Only the master data key is projected, regardless of what else the
	// user's Secret contains.
	if len(vol.Secret.Items) != 1 || vol.Secret.Items[0].Key != "master" || vol.Secret.Items[0].Path != "master" {
		t.Fatalf("expected items to project only the master key, got %+v", vol.Secret.Items)
	}
	if vol.Secret.DefaultMode == nil || *vol.Secret.DefaultMode != 0o400 {
		t.Fatalf("expected defaultMode 0400, got %v", vol.Secret.DefaultMode)
	}

	var mount *corev1.VolumeMount
	c := pod.Containers[0]
	for i := range c.VolumeMounts {
		if c.VolumeMounts[i].Name == l2EncryptionKeyVolumeName {
			mount = &c.VolumeMounts[i]
		}
	}
	if mount == nil {
		t.Fatal("expected l2-master-key volume mount")
	}
	if mount.MountPath != l2EncryptionKeyMountDir || !mount.ReadOnly {
		t.Fatalf("expected read-only mount at %s, got %+v", l2EncryptionKeyMountDir, mount)
	}
}

func TestBuildDaemonSet_NoSerdeNoMasterKeyMount(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
		RESP: &lmcachev1alpha1.RESPL2AdapterSpec{Host: "redis", Port: 6379},
	}

	ds := BuildDaemonSet(engine)
	for _, v := range ds.Spec.Template.Spec.Volumes {
		if v.Name == l2EncryptionKeyVolumeName {
			t.Fatal("unexpected l2-master-key volume without an aesgcm serde")
		}
	}
}

// ===========================
// BuildLookupService
// ===========================

func TestBuildLookupService_Default(t *testing.T) {
	engine := minimalEngine()
	svc := BuildLookupService(engine)

	if svc.Name != testEngineName {
		t.Fatalf("expected name test-engine, got %s", svc.Name)
	}
	if svc.Namespace != testNamespace {
		t.Fatalf("expected namespace default, got %s", svc.Namespace)
	}
	// Should NOT be headless — needs a ClusterIP for kube-proxy routing
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		t.Fatal("lookup service should not be headless")
	}
	// Should have internalTrafficPolicy=Local
	if svc.Spec.InternalTrafficPolicy == nil || *svc.Spec.InternalTrafficPolicy != corev1.ServiceInternalTrafficPolicyLocal {
		t.Fatal("expected internalTrafficPolicy=Local")
	}
	if len(svc.Spec.Ports) != 2 {
		t.Fatalf("expected 2 ports, got %d", len(svc.Spec.Ports))
	}
	if svc.Spec.Ports[0].Port != 5555 {
		t.Fatalf("expected server port 5555, got %d", svc.Spec.Ports[0].Port)
	}
	if svc.Spec.Ports[0].Name != serverPortName {
		t.Fatalf("expected port name server, got %s", svc.Spec.Ports[0].Name)
	}
	if svc.Spec.Ports[1].Port != 8080 {
		t.Fatalf("expected http port 8080, got %d", svc.Spec.Ports[1].Port)
	}
	if svc.Spec.Ports[1].Name != "http" {
		t.Fatalf("expected port name http, got %s", svc.Spec.Ports[1].Name)
	}

	sel := svc.Spec.Selector
	if sel["app.kubernetes.io/instance"] != testEngineName {
		t.Fatal("missing instance selector")
	}
}

func TestBuildLookupService_CustomPort(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Server = &lmcachev1alpha1.ServerSpec{
		Port: ptr(int32(8080)),
	}
	svc := BuildLookupService(engine)

	if svc.Spec.Ports[0].Port != 8080 {
		t.Fatalf("expected port 8080, got %d", svc.Spec.Ports[0].Port)
	}
}

// ===========================
// BuildMetricsService
// ===========================

func TestBuildMetricsService_Default(t *testing.T) {
	engine := minimalEngine()
	svc := BuildMetricsService(engine)

	if svc.Name != testEngineName+"-metrics" {
		t.Fatalf("expected name test-engine-metrics, got %s", svc.Name)
	}
	if svc.Namespace != testNamespace {
		t.Fatalf("expected namespace default, got %s", svc.Namespace)
	}
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Fatal("expected headless service (ClusterIP=None)")
	}
	if len(svc.Spec.Ports) != 1 {
		t.Fatalf("expected 1 port, got %d", len(svc.Spec.Ports))
	}
	if svc.Spec.Ports[0].Port != 9090 {
		t.Fatalf("expected port 9090, got %d", svc.Spec.Ports[0].Port)
	}
	if svc.Spec.Ports[0].Name != "metrics" {
		t.Fatalf("expected port name metrics, got %s", svc.Spec.Ports[0].Name)
	}

	// Selector should match SelectorLabels
	sel := svc.Spec.Selector
	if sel["app.kubernetes.io/instance"] != testEngineName {
		t.Fatal("missing instance selector")
	}
}

func TestBuildMetricsService_CustomPort(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Prometheus = &lmcachev1alpha1.PrometheusSpec{
		Port: ptr(int32(8888)),
	}
	svc := BuildMetricsService(engine)

	if svc.Spec.Ports[0].Port != 8888 {
		t.Fatalf("expected port 8888, got %d", svc.Spec.Ports[0].Port)
	}
}

// ===========================
// BuildConnectionConfigMap
// ===========================

func TestBuildConnectionConfigMap_Default(t *testing.T) {
	engine := minimalEngine()
	cm := BuildConnectionConfigMap(engine)

	if cm.Name != "test-engine-connection" {
		t.Fatalf("expected name test-engine-connection, got %s", cm.Name)
	}
	if cm.Namespace != testNamespace {
		t.Fatalf("expected namespace default, got %s", cm.Namespace)
	}

	jsonStr, ok := cm.Data["kv-transfer-config.json"]
	if !ok {
		t.Fatal("missing kv-transfer-config.json key")
	}

	var config map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &config); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if config["kv_connector"] != "LMCacheMPConnector" {
		t.Fatalf("expected kv_connector=LMCacheMPConnector, got %v", config["kv_connector"])
	}
	if config["kv_connector_module_path"] != "lmcache.integration.vllm.lmcache_mp_connector" {
		t.Fatalf(
			"expected kv_connector_module_path=lmcache.integration.vllm.lmcache_mp_connector, got %v",
			config["kv_connector_module_path"],
		)
	}
	if config["kv_role"] != "kv_both" {
		t.Fatalf("expected kv_role=kv_both, got %v", config["kv_role"])
	}

	extra := config["kv_connector_extra_config"].(map[string]any)
	if extra["lmcache.mp.host"] != "tcp://test-engine.default.svc.cluster.local" {
		t.Fatalf("expected service DNS host, got %v", extra["lmcache.mp.host"])
	}
	if extra["lmcache.mp.port"] != "5555" {
		t.Fatalf("expected port 5555, got %v", extra["lmcache.mp.port"])
	}
}

func TestBuildConnectionConfigMap_CustomPort(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Server = &lmcachev1alpha1.ServerSpec{
		Port: ptr(int32(8080)),
	}
	cm := BuildConnectionConfigMap(engine)

	jsonStr := cm.Data["kv-transfer-config.json"]
	var config map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &config); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	extra := config["kv_connector_extra_config"].(map[string]any)
	if extra["lmcache.mp.port"] != "8080" {
		t.Fatalf("expected port 8080, got %v", extra["lmcache.mp.port"])
	}
}

// ===========================
// ServiceMonitor
// ===========================

func TestServiceMonitorEnabled(t *testing.T) {
	tests := []struct {
		name string
		spec lmcachev1alpha1.LMCacheEngineSpec
		want bool
	}{
		{
			name: "nil prometheus",
			spec: lmcachev1alpha1.LMCacheEngineSpec{L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10}},
			want: false,
		},
		{
			name: "nil service monitor",
			spec: lmcachev1alpha1.LMCacheEngineSpec{
				L1:         lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
				Prometheus: &lmcachev1alpha1.PrometheusSpec{},
			},
			want: false,
		},
		{
			name: "disabled",
			spec: lmcachev1alpha1.LMCacheEngineSpec{
				L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
				Prometheus: &lmcachev1alpha1.PrometheusSpec{
					ServiceMonitor: &lmcachev1alpha1.ServiceMonitorSpec{Enabled: ptr(false)},
				},
			},
			want: false,
		},
		{
			name: "enabled",
			spec: lmcachev1alpha1.LMCacheEngineSpec{
				L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
				Prometheus: &lmcachev1alpha1.PrometheusSpec{
					ServiceMonitor: &lmcachev1alpha1.ServiceMonitorSpec{Enabled: ptr(true)},
				},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ServiceMonitorEnabled(&tt.spec)
			if got != tt.want {
				t.Fatalf("expected %v, got %v", tt.want, got)
			}
		})
	}
}

func TestBuildServiceMonitor(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Prometheus = &lmcachev1alpha1.PrometheusSpec{
		ServiceMonitor: &lmcachev1alpha1.ServiceMonitorSpec{
			Enabled:  ptr(true),
			Interval: ptr("15s"),
			Labels:   map[string]string{"release": "prom"},
		},
	}

	sm := BuildServiceMonitor(engine)

	if sm.Name != testEngineName {
		t.Fatalf("expected name test-engine, got %s", sm.Name)
	}
	if sm.Namespace != testNamespace {
		t.Fatalf("expected namespace default, got %s", sm.Namespace)
	}

	// Labels should include standard + custom
	if sm.Labels["release"] != "prom" {
		t.Fatal("missing custom label 'release'")
	}
	if sm.Labels["app.kubernetes.io/instance"] != testEngineName {
		t.Fatal("missing standard label")
	}

	// Endpoints
	if len(sm.Spec.Endpoints) != 1 {
		t.Fatalf("expected 1 endpoint, got %d", len(sm.Spec.Endpoints))
	}
	if string(sm.Spec.Endpoints[0].Interval) != "15s" {
		t.Fatalf("expected interval 15s, got %s", sm.Spec.Endpoints[0].Interval)
	}
	if sm.Spec.Endpoints[0].Port != "metrics" {
		t.Fatalf("expected port name metrics, got %s", sm.Spec.Endpoints[0].Port)
	}

	// Selector
	if sm.Spec.Selector.MatchLabels["app.kubernetes.io/instance"] != testEngineName {
		t.Fatal("missing selector label")
	}
}

// ===========================
// Test helpers
// ===========================

func assertArg(t *testing.T, args []string, flag, value string) {
	t.Helper()
	for i, a := range args {
		if a == flag {
			if i+1 < len(args) && args[i+1] == value {
				return
			}
			t.Fatalf("flag %s found but value is %s, expected %s", flag, args[i+1], value)
		}
	}
	t.Fatalf("flag %s not found in args: %v", flag, args)
}

func assertHasArg(t *testing.T, args []string, flag string) {
	t.Helper()
	if !slices.Contains(args, flag) {
		t.Fatalf("flag %s not found in args: %v", flag, args)
	}
}

func assertNoArg(t *testing.T, args []string, flag string) {
	t.Helper()
	for _, a := range args {
		if a == flag {
			t.Fatalf("flag %s should not be present in args", flag)
		}
	}
}

// --- GPUVendor tests ---

func TestBuildDaemonSet_GPUVendorNvidiaDefault(t *testing.T) {
	engine := minimalEngine()
	engine.SetDefaults()

	ds := BuildDaemonSet(engine)
	podSpec := ds.Spec.Template.Spec

	if podSpec.RuntimeClassName == nil || *podSpec.RuntimeClassName != nvidiaRuntimeClass {
		t.Fatalf("expected RuntimeClassName=nvidia, got %v", podSpec.RuntimeClassName)
	}

	c := podSpec.Containers[0]
	if !hasEnvAll(c.Env, "NVIDIA_VISIBLE_DEVICES") {
		t.Fatal("missing NVIDIA_VISIBLE_DEVICES=all on default (nvidia) vendor")
	}
	if !hasEnvAll(c.Env, "NVIDIA_DRIVER_CAPABILITIES") {
		t.Fatal("missing NVIDIA_DRIVER_CAPABILITIES=all on default (nvidia) vendor")
	}
}

func TestBuildDaemonSet_GPUVendorAMD(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.GPUVendor = ptr(lmcachev1alpha1.GPUVendorAMD)
	engine.SetDefaults()

	ds := BuildDaemonSet(engine)
	podSpec := ds.Spec.Template.Spec

	if podSpec.RuntimeClassName != nil {
		t.Fatalf("expected nil RuntimeClassName for AMD, got %q", *podSpec.RuntimeClassName)
	}

	c := podSpec.Containers[0]
	for _, e := range c.Env {
		if e.Name == "NVIDIA_VISIBLE_DEVICES" || e.Name == "NVIDIA_DRIVER_CAPABILITIES" {
			t.Fatalf("unexpected NVIDIA env var on AMD vendor: %s=%s", e.Name, e.Value)
		}
	}
}

func TestBuildDaemonSet_HostNetworkEnabled(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.HostNetwork = ptr(true)
	engine.SetDefaults()

	ds := BuildDaemonSet(engine)
	podSpec := ds.Spec.Template.Spec

	if !podSpec.HostNetwork {
		t.Fatal("expected HostNetwork=true")
	}
	if podSpec.DNSPolicy != corev1.DNSClusterFirstWithHostNet {
		t.Fatalf("expected DNSPolicy=ClusterFirstWithHostNet, got %s", podSpec.DNSPolicy)
	}
}

func TestBuildDaemonSet_PrivilegedDefaultFalse(t *testing.T) {
	// minimalEngine leaves spec.Privileged nil; the operator must not run the
	// container privileged unless explicitly opted in.
	ds := BuildDaemonSet(minimalEngine())
	c := ds.Spec.Template.Spec.Containers[0]

	if c.SecurityContext == nil || c.SecurityContext.Privileged == nil || *c.SecurityContext.Privileged {
		t.Fatal("expected privileged=false by default")
	}
}

func TestBuildDaemonSet_PrivilegedEnabled(t *testing.T) {
	engine := minimalEngine()
	engine.Spec.Privileged = ptr(true)

	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]

	if c.SecurityContext == nil || c.SecurityContext.Privileged == nil || !*c.SecurityContext.Privileged {
		t.Fatal("expected privileged=true when spec.privileged=true")
	}
}

// hasEnvAll reports whether envs contains an env var named name set to the
// literal "all" (the value the GPU passthrough vars are always set to).
func hasEnvAll(envs []corev1.EnvVar, name string) bool {
	for _, e := range envs {
		if e.Name == name && e.Value == "all" {
			return true
		}
	}
	return false
}
