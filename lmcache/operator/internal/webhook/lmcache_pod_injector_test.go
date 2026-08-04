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

package webhook

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

const (
	// testContainerVLLM is the default target container name in these specs.
	testContainerVLLM = "vllm"
	// testUnknownContainer names a container that does not exist on the pod.
	testUnknownContainer = "does-not-exist"
	// testPayloadRepo is the payload image repository used in staging specs.
	testPayloadRepo = "registry.example.com/lmcache/lmcache-payload"
	// testPayloadPullSecret is the engine's default payload pull secret name.
	testPayloadPullSecret = "lmcache-payload-pull"
)

// newLMCacheInjector returns an LMCachePodInjector backed by a fake client,
// seeded with the test engine's connection ConfigMap when seedConn is true. The
// LMCacheEngine CR itself is not seeded — the injector reads only the ConfigMap.
func newLMCacheInjector(seedConn bool) *LMCachePodInjector {
	scheme := newTestScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if seedConn {
		engine := &lmcachev1alpha1.LMCacheEngine{
			ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
		}
		builder = builder.WithRuntimeObjects(runtime.Object(resources.BuildConnectionConfigMap(engine)))
	}
	return &LMCachePodInjector{
		Client:  builder.Build(),
		Decoder: admission.NewDecoder(scheme),
	}
}

// lmcacheEngineWithInjection returns an LMCacheEngine whose name/namespace match
// the connection ConfigMap the injector reads. When payloadRepo is non-empty the
// engine carries an injection sub-spec with that payload image (tag "pinned") and
// a default payload pull secret; an empty payloadRepo yields injection=nil.
func lmcacheEngineWithInjection(payloadRepo string) *lmcachev1alpha1.LMCacheEngine {
	engine := &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
		Spec:       lmcachev1alpha1.LMCacheEngineSpec{L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10}},
	}
	if payloadRepo != "" {
		tag := "pinned"
		engine.Spec.Injection = &lmcachev1alpha1.LMCacheInjectionSpec{
			PayloadImage:     &lmcachev1alpha1.ImageSpec{Repository: &payloadRepo, Tag: &tag},
			ImagePullSecrets: []corev1.LocalObjectReference{{Name: testPayloadPullSecret}},
		}
	}
	return engine
}

// newLMCacheInjectorForEngine returns an injector seeded with the given engine CR
// and its connection ConfigMap. Unlike newLMCacheInjector this seeds the CR
// itself, exercising the injection sub-spec the injector reads.
func newLMCacheInjectorForEngine(engine *lmcachev1alpha1.LMCacheEngine) *LMCachePodInjector {
	scheme := newTestScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme).WithRuntimeObjects(
		engine, resources.BuildConnectionConfigMap(engine),
	)
	return &LMCachePodInjector{
		Client:  builder.Build(),
		Decoder: admission.NewDecoder(scheme),
	}
}

// findVolume returns the named volume from the pod, or nil.
func findVolume(pod *corev1.Pod, name string) *corev1.Volume {
	for i := range pod.Spec.Volumes {
		if pod.Spec.Volumes[i].Name == name {
			return &pod.Spec.Volumes[i]
		}
	}
	return nil
}

// findVolumeMount returns the named volume mount from the container, or nil.
func findVolumeMount(c *corev1.Container, name string) *corev1.VolumeMount {
	for i := range c.VolumeMounts {
		if c.VolumeMounts[i].Name == name {
			return &c.VolumeMounts[i]
		}
	}
	return nil
}

// lmcachePod returns a minimal args-only vLLM pod (no command override) opted in
// to LMCache injection via annotation. mutate may further customize it.
func lmcachePod(mutate func(*corev1.Pod)) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testPodName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				LMCacheAnnotationEngine: testEngineName,
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  testContainerVLLM,
					Image: "vllm/vllm-openai:latest",
					Args:  []string{"--model", "Qwen/Qwen2.5-0.5B"},
				},
			},
		},
	}
	if mutate != nil {
		mutate(pod)
	}
	return pod
}

var _ = Describe("LMCachePodInjector", func() {
	ctx := context.Background()

	Describe("full injection", func() {
		It("injects hostIPC, --kv-transfer-config, and PYTHONHASHSEED for an opted-in pod", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("hostIPC")
			Expect(out.Spec.HostIPC).To(BeTrue())

			c := findContainer(out, testContainerVLLM)
			Expect(c).NotTo(BeNil())

			By("--kv-transfer-config carries LMCacheMPConnector + tcp:// host")
			kv := argsFlagValue(c.Args, lmcFlagKVTransferConfig)
			Expect(kv).NotTo(BeEmpty())
			Expect(kv).To(ContainSubstring("LMCacheMPConnector"))
			Expect(kv).To(ContainSubstring(testSvcHost))

			By("the original --model arg is preserved")
			Expect(argsFlagValue(c.Args, "--model")).To(Equal("Qwen/Qwen2.5-0.5B"))

			By("PYTHONHASHSEED set to 0")
			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))

			By("idempotency annotation stamped, no skip reason")
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationSkipReason))
		})

		It("respects a user-set PYTHONHASHSEED", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Env = []corev1.EnvVar{
					{Name: pythonHashSeedEnvName, Value: "42"},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, testContainerVLLM)

			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal("42"))
		})
	})

	Describe("gating", func() {
		It("allows a pod with no engine annotation unchanged", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				delete(p.Annotations, LMCacheAnnotationEngine)
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("allows an already-injected pod as a no-op", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationInjected] = valueTrue
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("skips + stamps engine-not-found when the connection ConfigMap is absent", func() {
			injector := newLMCacheInjector(false) // no connection ConfigMap seeded
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonEngineNotFound))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})

		It("skips + stamps command-override when the target container overrides command", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exec vllm serve"}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonCommandOverride))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})

		It("skips + stamps target-container-not-found for an unknown container name", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationContainer] = testUnknownContainer
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonTargetContainerNotFound))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})
	})

	Describe("user-supplied --kv-transfer-config", func() {
		It("skips + stamps but still applies hostIPC + PYTHONHASHSEED", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Args = []string{"--kv-transfer-config", `{"kv_connector":"Other"}`}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, testContainerVLLM)

			By("the user's kv-transfer-config JSON is untouched")
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).To(Equal(`{"kv_connector":"Other"}`))
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).NotTo(ContainSubstring("LMCacheMPConnector"))

			By("the skip reason is stamped but the rest of the injection still applies")
			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonKVTransferConfigPresent))
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))
		})
	})

	Describe("target container resolution", func() {
		It("injects into the annotation-named non-first container", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationContainer] = testContainerVLLM
				p.Spec.Containers = []corev1.Container{
					{Name: "sidecar", Image: "busybox", Args: []string{"sleep"}},
					{Name: testContainerVLLM, Image: "vllm/vllm-openai:latest", Args: []string{"--model", "m"}},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("the vLLM container is mutated")
			vllm := findContainer(out, testContainerVLLM)
			Expect(argsFlagValue(vllm.Args, lmcFlagKVTransferConfig)).To(ContainSubstring("LMCacheMPConnector"))
			Expect(envValue(vllm, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))

			By("the sidecar container is untouched")
			sidecar := findContainer(out, "sidecar")
			Expect(sidecar.Args).To(Equal([]string{"sleep"}))
			Expect(envValue(sidecar, pythonHashSeedEnvName)).To(BeEmpty())
		})
	})

	Describe("code-payload staging", func() {
		It("stages the payload (volume + init + mount + PYTHONPATH + secret) and still wires the connection", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(testPayloadRepo))
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("emptyDir payload volume")
			vol := findVolume(out, lmcPayloadVolumeName)
			Expect(vol).NotTo(BeNil())
			Expect(vol.EmptyDir).NotTo(BeNil())

			By("payload init container with image + SHARED_DIR + read-write mount")
			Expect(out.Spec.InitContainers).To(HaveLen(1))
			init := out.Spec.InitContainers[0]
			Expect(init.Name).To(Equal(lmcPayloadInitName))
			Expect(init.Image).To(Equal(testPayloadRepo + ":pinned"))
			Expect(envValue(&init, lmcSharedDirEnvName)).To(Equal(lmcPayloadMountPath))
			By("no command override — the payload image's own busybox cp ENTRYPOINT runs")
			Expect(init.Command).To(BeEmpty())
			Expect(init.ImagePullPolicy).To(Equal(corev1.PullIfNotPresent))
			Expect(init.VolumeMounts).To(HaveLen(1))
			Expect(init.VolumeMounts[0].Name).To(Equal(lmcPayloadVolumeName))
			Expect(init.VolumeMounts[0].MountPath).To(Equal(lmcPayloadMountPath))
			Expect(init.VolumeMounts[0].ReadOnly).To(BeFalse())

			c := findContainer(out, testContainerVLLM)
			By("read-only mount on the vLLM container")
			mount := findVolumeMount(c, lmcPayloadVolumeName)
			Expect(mount).NotTo(BeNil())
			Expect(mount.ReadOnly).To(BeTrue())
			Expect(mount.MountPath).To(Equal(lmcPayloadMountPath))

			By("PYTHONPATH set to the payload mount path")
			Expect(envValue(c, pythonPathEnvName)).To(Equal(lmcPayloadMountPath))

			By("the engine's payload pull secret is merged onto the pod")
			Expect(out.Spec.ImagePullSecrets).To(ContainElement(corev1.LocalObjectReference{Name: testPayloadPullSecret}))

			By("connection wiring still applied")
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).To(ContainSubstring("LMCacheMPConnector"))
			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
		})

		It("does not stage when injection.payloadImage is unset (connection-only)", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(""))
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Spec.InitContainers).To(BeEmpty())
			Expect(findVolume(out, lmcPayloadVolumeName)).To(BeNil())
			c := findContainer(out, testContainerVLLM)
			Expect(envValue(c, pythonPathEnvName)).To(BeEmpty())

			By("connection wiring still applied")
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).To(ContainSubstring("LMCacheMPConnector"))
		})

		It("does not stage when the engine CR is absent (connection-only via ConfigMap)", func() {
			injector := newLMCacheInjector(true) // seeds only the connection ConfigMap, not the CR
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Spec.InitContainers).To(BeEmpty())
			Expect(findVolume(out, lmcPayloadVolumeName)).To(BeNil())
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
		})

		It("prepends to a pre-existing PYTHONPATH", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(testPayloadRepo))
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Env = []corev1.EnvVar{{Name: pythonPathEnvName, Value: "/opt/extra"}}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, testContainerVLLM)

			Expect(envValue(c, pythonPathEnvName)).To(Equal(lmcPayloadMountPath + ":/opt/extra"))
		})

		It("honors the image-pull-secrets annotation override", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(testPayloadRepo))
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationImagePullSecrets] = "override-a,override-b"
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Spec.ImagePullSecrets).To(ContainElement(corev1.LocalObjectReference{Name: "override-a"}))
			Expect(out.Spec.ImagePullSecrets).To(ContainElement(corev1.LocalObjectReference{Name: "override-b"}))

			By("the annotation overrides the engine default rather than appending to it")
			Expect(out.Spec.ImagePullSecrets).NotTo(ContainElement(corev1.LocalObjectReference{Name: testPayloadPullSecret}))
		})

		It("does not duplicate a payload pull secret the pod already lists", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(testPayloadRepo))
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: testPayloadPullSecret}}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			count := 0
			for _, s := range out.Spec.ImagePullSecrets {
				if s.Name == testPayloadPullSecret {
					count++
				}
			}
			Expect(count).To(Equal(1), "the engine's payload pull secret must be merged, not duplicated")
		})

		It("uses the engine injection.targetContainer default to select a non-first container", func() {
			engine := lmcacheEngineWithInjection(testPayloadRepo)
			targetName := testContainerVLLM
			engine.Spec.Injection.TargetContainer = &targetName
			injector := newLMCacheInjectorForEngine(engine)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers = []corev1.Container{
					{Name: "sidecar", Image: "busybox", Args: []string{"sleep"}},
					{Name: testContainerVLLM, Image: "vllm/vllm-openai:latest", Args: []string{"--model", "m"}},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("the engine-named vLLM container is wired + staged")
			vllm := findContainer(out, testContainerVLLM)
			Expect(argsFlagValue(vllm.Args, lmcFlagKVTransferConfig)).To(ContainSubstring("LMCacheMPConnector"))
			Expect(envValue(vllm, pythonPathEnvName)).To(Equal(lmcPayloadMountPath))
			Expect(findVolumeMount(vllm, lmcPayloadVolumeName)).NotTo(BeNil())

			By("the first (sidecar) container is untouched")
			sidecar := findContainer(out, "sidecar")
			Expect(sidecar.Args).To(Equal([]string{"sleep"}))
			Expect(findVolumeMount(sidecar, lmcPayloadVolumeName)).To(BeNil())
		})

		It("is idempotent: pre-staged volume + init container are not duplicated", func() {
			injector := newLMCacheInjectorForEngine(lmcacheEngineWithInjection(testPayloadRepo))
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Volumes = []corev1.Volume{lmcStaging.volume()}
				p.Spec.InitContainers = []corev1.Container{
					lmcStaging.initContainer(testPayloadRepo+":pinned", corev1.PullIfNotPresent),
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Spec.Volumes).To(HaveLen(1))
			Expect(out.Spec.InitContainers).To(HaveLen(1))
		})
	})
})
