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
	"testing"
)

// ptr is defined in lmcacheengine_test.go (same package); reuse it here.

// --- SetDefaults tests ---

func TestCBSetDefaults_LogLevelNil(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.LogLevel == nil || *e.Spec.LogLevel != defaultLogLevel {
		t.Fatalf("expected LogLevel=INFO, got %v", e.Spec.LogLevel)
	}
}

func TestCBSetDefaults_LogLevelPreserved(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:       L1BackendSpec{SizeGB: 10},
		LogLevel: ptr("DEBUG"),
	}}
	e.SetDefaults()
	if *e.Spec.LogLevel != "DEBUG" {
		t.Fatalf("expected LogLevel=DEBUG, got %s", *e.Spec.LogLevel)
	}
}

func TestCBSetDefaults_NodeSelectorDefaultGPU(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.NodeSelector == nil {
		t.Fatal("expected default NodeSelector, got nil")
	}
	if e.Spec.NodeSelector["nvidia.com/gpu.present"] != labelValueTrue {
		t.Fatalf("expected nvidia.com/gpu.present=true, got %v", e.Spec.NodeSelector)
	}
}

func TestCBSetDefaults_NodeSelectorPreserved(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:           L1BackendSpec{SizeGB: 10},
		NodeSelector: map[string]string{"custom": "label"},
	}}
	e.SetDefaults()
	if e.Spec.NodeSelector["custom"] != "label" {
		t.Fatal("expected custom node selector preserved")
	}
	if _, ok := e.Spec.NodeSelector["nvidia.com/gpu.present"]; ok {
		t.Fatal("default should not override user-provided NodeSelector")
	}
}

func TestCBSetDefaults_GPUVendorDefaultNvidia(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.GPUVendor == nil || *e.Spec.GPUVendor != GPUVendorNvidia {
		t.Fatalf("expected GPUVendor=nvidia, got %v", e.Spec.GPUVendor)
	}
}

func TestCBSetDefaults_GPUVendorAMDSkipsNodeSelectorDefault(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		GPUVendor: ptr(GPUVendorAMD),
	}}
	e.SetDefaults()
	if e.Spec.NodeSelector != nil {
		t.Fatalf("expected nil NodeSelector for AMD vendor, got %v", e.Spec.NodeSelector)
	}
}

func TestCBSetDefaults_ChunkSizeDefault256(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.Server == nil || e.Spec.Server.ChunkSize == nil {
		t.Fatal("expected Server.ChunkSize to be defaulted, got nil")
	}
	if *e.Spec.Server.ChunkSize != CacheBlendChunkSize {
		t.Fatalf("expected ChunkSize=256, got %d", *e.Spec.Server.ChunkSize)
	}
}

func TestCBSetDefaults_ChunkSizePreserved(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:     L1BackendSpec{SizeGB: 10},
		Server: &ServerSpec{ChunkSize: ptr(int32(512))},
	}}
	e.SetDefaults()
	if *e.Spec.Server.ChunkSize != 512 {
		t.Fatalf("expected ChunkSize preserved at 512, got %d", *e.Spec.Server.ChunkSize)
	}
}

func TestCBSetDefaults_BlendDefaults(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.Blend == nil {
		t.Fatal("expected Blend to be defaulted, got nil")
	}
	if e.Spec.Blend.CheckLayer == nil || *e.Spec.Blend.CheckLayer != 1 {
		t.Fatalf("expected CheckLayer=1, got %v", e.Spec.Blend.CheckLayer)
	}
	if e.Spec.Blend.RecompRatio == nil || *e.Spec.Blend.RecompRatio != 0.15 {
		t.Fatalf("expected RecompRatio=0.15, got %v", e.Spec.Blend.RecompRatio)
	}
}

func TestCBSetDefaults_BlendPreserved(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:    L1BackendSpec{SizeGB: 10},
		Blend: &BlendSpec{CheckLayer: ptr(int32(3)), RecompRatio: ptr(0.5)},
	}}
	e.SetDefaults()
	if *e.Spec.Blend.CheckLayer != 3 {
		t.Fatalf("expected CheckLayer preserved at 3, got %d", *e.Spec.Blend.CheckLayer)
	}
	if *e.Spec.Blend.RecompRatio != 0.5 {
		t.Fatalf("expected RecompRatio preserved at 0.5, got %v", *e.Spec.Blend.RecompRatio)
	}
}

func TestCBSetDefaults_InjectionDefaults(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.Injection == nil {
		t.Fatal("expected Injection to be defaulted, got nil")
	}
	if e.Spec.Injection.Cudagraph == nil || *e.Spec.Injection.Cudagraph != CudagraphEager {
		t.Fatalf("expected Cudagraph=eager, got %v", e.Spec.Injection.Cudagraph)
	}
	// payloadImage was not set, so it stays nil (no cluster-wide default).
	if e.Spec.Injection.PayloadImage != nil {
		t.Fatalf("expected PayloadImage nil when unset, got %v", e.Spec.Injection.PayloadImage)
	}

	// When payloadImage is set, tag and pullPolicy default.
	e2 := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: &InjectionSpec{PayloadImage: &ImageSpec{Repository: ptr("myreg/cacheblend-plugin")}},
	}}
	e2.SetDefaults()
	pi := e2.Spec.Injection.PayloadImage
	if pi.Tag == nil || *pi.Tag != "latest" {
		t.Fatalf("expected payloadImage.Tag=latest, got %v", pi.Tag)
	}
	if pi.PullPolicy == nil || *pi.PullPolicy != "IfNotPresent" {
		t.Fatalf("expected payloadImage.PullPolicy=IfNotPresent, got %v", pi.PullPolicy)
	}
}

func TestCBSetDefaults_InjectionPreserved(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1: L1BackendSpec{SizeGB: 10},
		Injection: &InjectionSpec{
			PayloadImage: &ImageSpec{Repository: ptr("myreg/cacheblend-plugin"), Tag: ptr("v1"), PullPolicy: ptr("Always")},
			Cudagraph:    ptr(CudagraphFullDecodeOnly),
		},
	}}
	e.SetDefaults()
	if *e.Spec.Injection.PayloadImage.PullPolicy != "Always" {
		t.Fatalf("expected payloadImage.PullPolicy preserved at Always, got %s", *e.Spec.Injection.PayloadImage.PullPolicy)
	}
	if *e.Spec.Injection.PayloadImage.Tag != "v1" {
		t.Fatalf("expected payloadImage.Tag preserved at v1, got %s", *e.Spec.Injection.PayloadImage.Tag)
	}
	if *e.Spec.Injection.Cudagraph != CudagraphFullDecodeOnly {
		t.Fatalf("expected Cudagraph preserved at full_decode_only, got %s", *e.Spec.Injection.Cudagraph)
	}
}

// --- ValidateSpec tests ---

// validCBInjection returns the minimal injection block ValidateSpec requires
// (injection.payloadImage.repository). Tests that exercise other fields include
// it so the injection requirement does not perturb their expected error counts.
func validCBInjection() *InjectionSpec {
	return &InjectionSpec{PayloadImage: &ImageSpec{Repository: ptr("myreg/cacheblend-plugin")}}
}

func TestCBValidateSpec_ValidMinimal(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
	}}
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got %v", errs)
	}
}

func TestCBValidateSpec_ValidDefaulted(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
	}}
	e.SetDefaults()
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors after SetDefaults, got %v", errs)
	}
}

func TestCBValidateSpec_InjectionRequired(t *testing.T) {
	// A spec with no injection block is invalid: the webhook has no payload
	// image to inject.
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0].Field != "spec.injection" {
		t.Fatalf("expected field spec.injection, got %s", errs[0].Field)
	}
}

func TestCBValidateSpec_InjectionPayloadRepositoryRequired(t *testing.T) {
	// injection present but payloadImage.repository empty is invalid.
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: &InjectionSpec{PayloadImage: &ImageSpec{Repository: ptr("")}},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0].Field != "spec.injection.payloadImage.repository" {
		t.Fatalf("expected field spec.injection.payloadImage.repository, got %s", errs[0].Field)
	}
}

func TestCBValidateSpec_SizeGBZero(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 0},
		Injection: validCBInjection(),
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0].Field != "spec.l1.sizeGB" {
		t.Fatalf("expected field spec.l1.sizeGB, got %s", errs[0].Field)
	}
}

func TestCBValidateSpec_SizeGBNegative(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: -1},
		Injection: validCBInjection(),
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d", len(errs))
	}
}

func TestCBValidateSpec_EvictionPolicyInvalid(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		Eviction:  &EvictionSpec{Policy: ptr("FIFO")},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}

func TestCBValidateSpec_EvictionPolicyLRU(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		Eviction:  &EvictionSpec{Policy: ptr("LRU")},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got %v", errs)
	}
}

func TestCBValidateSpec_TriggerWatermarkBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   float64
		wantErr bool
	}{
		{"zero", 0.0, true},
		{"negative", -0.1, true},
		{"valid_low", 0.01, false},
		{"valid_mid", 0.5, false},
		{"valid_one", 1.0, false},
		{"above_one", 1.1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Eviction:  &EvictionSpec{TriggerWatermark: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_EvictionRatioBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   float64
		wantErr bool
	}{
		{"zero", 0.0, true},
		{"negative", -0.5, true},
		{"valid_low", 0.1, false},
		{"valid_one", 1.0, false},
		{"above_one", 1.5, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Eviction:  &EvictionSpec{EvictionRatio: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_ServerPort(t *testing.T) {
	tests := []struct {
		name    string
		port    int32
		wantErr bool
	}{
		{"below_min", 80, true},
		{"min_valid", 1024, false},
		{"max_valid", 65535, false},
		{"above_max", 65536, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Server:    &ServerSpec{Port: ptr(tt.port), ChunkSize: ptr(CacheBlendChunkSize)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_ChunkSize(t *testing.T) {
	tests := []struct {
		name    string
		value   int32
		wantErr bool
	}{
		{"valid_256", 256, false},
		{"too_small", 128, true},
		{"too_large", 512, true},
		{"zero", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Server:    &ServerSpec{ChunkSize: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_CheckLayerBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   int32
		wantErr bool
	}{
		{"zero", 0, false},
		{"positive", 5, false},
		{"negative", -1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Blend:     &BlendSpec{CheckLayer: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_RecompRatioBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   float64
		wantErr bool
	}{
		{"zero", 0.0, true},
		{"negative", -0.1, true},
		{"valid_low", 0.01, false},
		{"valid_default", 0.15, false},
		{"valid_one", 1.0, false},
		{"above_one", 1.1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
				L1:        L1BackendSpec{SizeGB: 10},
				Injection: validCBInjection(),
				Blend:     &BlendSpec{RecompRatio: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestCBValidateSpec_MultipleErrors(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 0},
		Injection: validCBInjection(),
		Server:    &ServerSpec{Port: ptr(int32(80)), ChunkSize: ptr(int32(128))},
		Eviction: &EvictionSpec{
			Policy:           ptr("FIFO"),
			TriggerWatermark: ptr(0.0),
			EvictionRatio:    ptr(0.0),
		},
		Blend: &BlendSpec{CheckLayer: ptr(int32(-1)), RecompRatio: ptr(2.0)},
	}}
	errs := e.ValidateSpec()
	// sizeGB, port, chunkSize, policy, watermark, ratio, checkLayer, recompRatio = 8 errors
	if len(errs) != 8 {
		t.Fatalf("expected 8 errors, got %d: %v", len(errs), errs)
	}
}

// --- L2 Backend validation tests ---

func TestCBValidateSpec_L2RESPValid(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		L2Backend: &L2BackendSpec{
			RESP: &RESPL2AdapterSpec{
				Host: "redis.default.svc",
				Port: 6379,
			},
		},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got %v", errs)
	}
}

func TestCBValidateSpec_L2RESPEmptyHost(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		L2Backend: &L2BackendSpec{
			RESP: &RESPL2AdapterSpec{
				Host: "",
				Port: 6379,
			},
		},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0].Field != "spec.l2Backend.resp.host" {
		t.Fatalf("expected field spec.l2Backend.resp.host, got %s", errs[0].Field)
	}
}

func TestCBValidateSpec_L2NoneSet(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		L2Backend: &L2BackendSpec{},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}

func TestCBValidateSpec_L2BothSet(t *testing.T) {
	e := &CacheBlendEngine{Spec: CacheBlendEngineSpec{
		L1:        L1BackendSpec{SizeGB: 10},
		Injection: validCBInjection(),
		L2Backend: &L2BackendSpec{
			RESP: &RESPL2AdapterSpec{Host: "redis", Port: 6379},
			Raw:  &RawL2AdapterSpec{Type: "mock"},
		},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}
