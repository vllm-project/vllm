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
	"slices"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

func minimalCoordinator() *lmcachev1alpha1.LMCacheCoordinator {
	return &lmcachev1alpha1.LMCacheCoordinator{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-coordinator",
			Namespace: testNamespace,
		},
		Spec: lmcachev1alpha1.LMCacheCoordinatorSpec{},
	}
}

func TestBuildCoordinatorArgs_Defaults(t *testing.T) {
	args := BuildCoordinatorArgs(&minimalCoordinator().Spec)

	if got := findArgValue(t, args, "--host"); got != "0.0.0.0" {
		t.Errorf("--host = %q, want 0.0.0.0", got)
	}
	if got := findArgValue(t, args, "--port"); got != "9300" {
		t.Errorf("--port = %q, want 9300", got)
	}
	if got := findArgValue(t, args, "--instance-timeout"); got != "30" {
		t.Errorf("--instance-timeout = %q, want 30", got)
	}
	if got := findArgValue(t, args, "--eviction-ratio"); got != "0.2" {
		t.Errorf("--eviction-ratio = %q, want 0.2", got)
	}
	// Unset blend knobs are omitted so the coordinator image applies its own
	// defaults; emitting them would break images whose CLI predates the flags.
	if slices.Contains(args, "--blend-chunk-size") {
		t.Errorf("--blend-chunk-size should be omitted when unset, got args %v", args)
	}
	if slices.Contains(args, "--blend-probe-stride") {
		t.Errorf("--blend-probe-stride should be omitted when unset, got args %v", args)
	}
}

func TestBuildCoordinatorArgs_Overrides(t *testing.T) {
	c := minimalCoordinator()
	c.Spec.Port = ptr(int32(9400))
	c.Spec.EvictionRatio = ptr(0.5)
	c.Spec.BlendChunkSize = ptr(int32(512))
	c.Spec.BlendProbeStride = ptr(int32(4))
	c.Spec.ExtraArgs = []string{"--port", "1234"}

	args := BuildCoordinatorArgs(&c.Spec)

	if got := findArgValue(t, args, "--eviction-ratio"); got != "0.5" {
		t.Errorf("--eviction-ratio = %q, want 0.5", got)
	}
	if got := findArgValue(t, args, "--blend-chunk-size"); got != "512" {
		t.Errorf("--blend-chunk-size = %q, want 512", got)
	}
	if got := findArgValue(t, args, "--blend-probe-stride"); got != "4" {
		t.Errorf("--blend-probe-stride = %q, want 4", got)
	}
	// ExtraArgs are appended last so they win when re-specifying a flag.
	if args[len(args)-2] != "--port" || args[len(args)-1] != "1234" {
		t.Errorf("expected extraArgs appended last, got tail %v", args[len(args)-2:])
	}
}

func TestBuildCoordinatorDeployment_Basics(t *testing.T) {
	c := minimalCoordinator()
	c.Spec.Replicas = ptr(int32(2))
	deploy := BuildCoordinatorDeployment(c)

	if deploy.Spec.Replicas == nil || *deploy.Spec.Replicas != 2 {
		t.Errorf("expected replicas 2 to pass through to the Deployment")
	}

	podSpec := deploy.Spec.Template.Spec
	if len(podSpec.Containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(podSpec.Containers))
	}
	container := podSpec.Containers[0]
	if !slices.Contains(container.Command, "coordinator") {
		t.Errorf("expected coordinator subcommand in command %v", container.Command)
	}
	if container.ReadinessProbe == nil || container.ReadinessProbe.HTTPGet == nil ||
		container.ReadinessProbe.HTTPGet.Path != "/healthz" {
		t.Errorf("expected readiness probe on /healthz")
	}
	if len(container.Ports) != 1 || container.Ports[0].ContainerPort != 9300 {
		t.Errorf("expected single container port 9300, got %v", container.Ports)
	}
	// No GPU / privileged scaffolding for the coordinator.
	if podSpec.RuntimeClassName != nil {
		t.Errorf("coordinator should not set a runtime class")
	}
}

func TestBuildCoordinatorService_ClusterIP(t *testing.T) {
	c := minimalCoordinator()
	svc := BuildCoordinatorService(c)

	if svc.Name != "test-coordinator" {
		t.Errorf("service name = %q, want test-coordinator", svc.Name)
	}
	if len(svc.Spec.Ports) != 1 || svc.Spec.Ports[0].Port != 9300 {
		t.Errorf("expected single service port 9300, got %v", svc.Spec.Ports)
	}
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		t.Errorf("main coordinator service must not be headless")
	}
}

func TestCoordinatorEndpoint(t *testing.T) {
	c := minimalCoordinator()
	c.Spec.Port = ptr(int32(9400))
	want := "http://test-coordinator.default.svc:9400"
	if got := CoordinatorEndpoint(c); got != want {
		t.Errorf("CoordinatorEndpoint = %q, want %q", got, want)
	}
}

func TestBuildContainerArgs_CoordinatorURL(t *testing.T) {
	engine := minimalEngine()
	url := "http://my-coordinator.default.svc:9300"
	engine.Spec.Coordinator = &lmcachev1alpha1.CoordinatorConnectionSpec{
		URL:              ptr(url),
		L2EventReporting: ptr(true),
	}

	args := BuildContainerArgs(&engine.Spec)

	if got := findArgValue(t, args, "--coordinator-url"); got != url {
		t.Errorf("--coordinator-url = %q, want %q", got, url)
	}
	if got := findArgValue(t, args, "--coordinator-heartbeat-interval"); got != "5" {
		t.Errorf("--coordinator-heartbeat-interval = %q, want 5", got)
	}
	if !slices.Contains(args, "--coordinator-l2-event-reporting") {
		t.Errorf("expected --coordinator-l2-event-reporting flag")
	}
}

func TestBuildContainerArgs_CoordinatorDisabledWithoutURL(t *testing.T) {
	engine := minimalEngine()
	// Only a ref is set (unresolved); the builder must not emit a malformed flag.
	engine.Spec.Coordinator = &lmcachev1alpha1.CoordinatorConnectionSpec{
		Ref: &corev1.LocalObjectReference{Name: "my-coordinator"},
	}

	args := BuildContainerArgs(&engine.Spec)
	if slices.Contains(args, "--coordinator-url") {
		t.Errorf("expected no --coordinator-url when URL unresolved, got %v", args)
	}
}
