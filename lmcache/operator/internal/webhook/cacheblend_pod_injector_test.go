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
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	jsonpatch "github.com/evanphx/json-patch/v5"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

const (
	testEngineName = "cb"
	testNamespace  = "vllm-ns"
	testPodName    = "vllm-pod"
	testSvcHost    = "tcp://cb.vllm-ns.svc.cluster.local"
)

// newTestScheme returns a scheme with clientgo + the lmcache v1alpha1 types.
func newTestScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	Expect(clientgoscheme.AddToScheme(s)).To(Succeed())
	Expect(lmcachev1alpha1.AddToScheme(s)).To(Succeed())
	return s
}

// newTestEngine returns a defaulted CacheBlendEngine with the given injection
// overrides applied via mutate (nil = pure defaults).
func newTestEngine(mutate func(*lmcachev1alpha1.CacheBlendEngine)) *lmcachev1alpha1.CacheBlendEngine {
	payloadRepo := "registry.example.com/lmcache/cacheblend-payload"
	payloadTag := "pinned"
	engine := &lmcachev1alpha1.CacheBlendEngine{
		ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
		Spec: lmcachev1alpha1.CacheBlendEngineSpec{
			L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
			Injection: &lmcachev1alpha1.InjectionSpec{
				PayloadImage: &lmcachev1alpha1.ImageSpec{Repository: &payloadRepo, Tag: &payloadTag},
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "cb-payload-pull"},
				},
			},
		},
	}
	engine.SetDefaults()
	if mutate != nil {
		mutate(engine)
	}
	return engine
}

// newPodInjector returns a CacheBlendPodInjector backed by a fake client seeded
// with the given objects, plus the engine's connection ConfigMap when seedConn
// is true.
func newPodInjector(
	engine *lmcachev1alpha1.CacheBlendEngine,
	seedConn bool,
) *CacheBlendPodInjector {
	scheme := newTestScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	objs := []runtime.Object{engine}
	if seedConn {
		objs = append(objs, resources.BuildCBConnectionConfigMap(engine))
	}
	builder = builder.WithRuntimeObjects(objs...)
	return &CacheBlendPodInjector{
		Client:  builder.Build(),
		Decoder: admission.NewDecoder(scheme),
	}
}

// makeRequest builds a CREATE admission.Request carrying the given pod as raw
// JSON in req.Object.
func makeRequest(pod *corev1.Pod) admission.Request {
	pod.TypeMeta = metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}
	if pod.Namespace == "" {
		pod.Namespace = testNamespace
	}
	raw, err := json.Marshal(pod)
	Expect(err).NotTo(HaveOccurred())
	return admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: pod.Namespace,
			Object:    runtime.RawExtension{Raw: raw},
		},
	}
}

// applyResponse applies the response's JSON patches to the original pod JSON and
// returns the mutated pod. It asserts the response is Allowed.
func applyResponse(original *corev1.Pod, resp admission.Response) *corev1.Pod {
	Expect(resp.Allowed).To(BeTrue(), "expected the response to be Allowed")

	origRaw, err := json.Marshal(original)
	Expect(err).NotTo(HaveOccurred())

	if len(resp.Patches) == 0 {
		out := &corev1.Pod{}
		Expect(json.Unmarshal(origRaw, out)).To(Succeed())
		return out
	}

	patchRaw, err := json.Marshal(resp.Patches)
	Expect(err).NotTo(HaveOccurred())
	patch, err := jsonpatch.DecodePatch(patchRaw)
	Expect(err).NotTo(HaveOccurred())
	mutatedRaw, err := patch.Apply(origRaw)
	Expect(err).NotTo(HaveOccurred())

	out := &corev1.Pod{}
	Expect(json.Unmarshal(mutatedRaw, out)).To(Succeed())
	return out
}

// vllmPod returns a minimal args-only vLLM pod (no command override) bound to
// the test engine via annotation. mutate may further customize it.
func vllmPod(mutate func(*corev1.Pod)) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testPodName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				AnnotationEngine: testEngineName,
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "vllm",
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

// findContainer returns the named container from the pod, or nil.
func findContainer(pod *corev1.Pod, name string) *corev1.Container {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			return &pod.Spec.Containers[i]
		}
	}
	return nil
}

// envValue returns the value of the named env var on the container, or "".
func envValue(c *corev1.Container, name string) string {
	for _, e := range c.Env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

// pullSecretNames returns the names in the pod's imagePullSecrets.
func pullSecretNames(pod *corev1.Pod) []string {
	out := make([]string, 0, len(pod.Spec.ImagePullSecrets))
	for _, s := range pod.Spec.ImagePullSecrets {
		out = append(out, s.Name)
	}
	return out
}

var _ = Describe("CacheBlendPodInjector", func() {
	ctx := context.Background()

	Describe("full M0–M7 injection", func() {
		It("injects all required mutations for an opted-in pod", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("M0: hostIPC")
			Expect(out.Spec.HostIPC).To(BeTrue())

			By("M1: cb-plugin emptyDir volume")
			var vol *corev1.Volume
			for i := range out.Spec.Volumes {
				if out.Spec.Volumes[i].Name == cbPluginVolumeName {
					vol = &out.Spec.Volumes[i]
				}
			}
			Expect(vol).NotTo(BeNil())
			Expect(vol.EmptyDir).NotTo(BeNil())

			By("M2: payload init container with image + pull policy + SHARED_DIR")
			Expect(out.Spec.InitContainers).To(HaveLen(1))
			init := out.Spec.InitContainers[0]
			Expect(init.Image).To(Equal("registry.example.com/lmcache/cacheblend-payload:pinned"))
			Expect(init.ImagePullPolicy).To(Equal(corev1.PullIfNotPresent))
			Expect(init.Command).To(BeEmpty())
			Expect(envValue(&init, cbSharedDirEnvName)).To(Equal(cbPluginMountPath))
			Expect(init.VolumeMounts).To(HaveLen(1))
			Expect(init.VolumeMounts[0].Name).To(Equal(cbPluginVolumeName))
			Expect(init.VolumeMounts[0].MountPath).To(Equal(cbPluginMountPath))
			Expect(init.VolumeMounts[0].ReadOnly).To(BeFalse())

			c := findContainer(out, "vllm")
			Expect(c).NotTo(BeNil())

			By("M3: read-only mount on the target container")
			var mount *corev1.VolumeMount
			for i := range c.VolumeMounts {
				if c.VolumeMounts[i].Name == cbPluginVolumeName {
					mount = &c.VolumeMounts[i]
				}
			}
			Expect(mount).NotTo(BeNil())
			Expect(mount.ReadOnly).To(BeTrue())
			Expect(mount.MountPath).To(Equal(cbPluginMountPath))

			By("M4: PYTHONPATH on the container")
			Expect(envValue(c, pythonPathEnvName)).To(Equal(cbPluginMountPath))

			By("M5: required vLLM args asserted individually")
			Expect(argsHasFlagValue(c.Args, cbFlagAttentionBackend, cbValAttentionBackend)).To(BeTrue(),
				"--attention-backend=CUSTOM")
			Expect(argsHasFlag(c.Args, cbFlagNoChunkedPrefill)).To(BeTrue(),
				"--no-enable-chunked-prefill")
			Expect(argsHasFlagValue(c.Args, cbFlagBlockSize, cbValBlockSize)).To(BeTrue(),
				"--block-size=64")
			Expect(argsHasFlagValue(c.Args, cbFlagPipelineParallelSize, cbValPipelineParallelSize)).To(BeTrue(),
				"--pipeline-parallel-size=1")
			Expect(argsHasFlag(c.Args, cbFlagNoAsyncScheduling)).To(BeTrue(),
				"--no-async-scheduling")
			Expect(argsHasFlag(c.Args, cbFlagEnforceEager)).To(BeTrue(),
				"default cudagraph eager -> --enforce-eager")

			By("M5: --kv-transfer-config carries CBKVConnector + tcp:// host")
			kv := argsFlagValue(c.Args, cbFlagKVTransferConfig)
			Expect(kv).NotTo(BeEmpty())
			Expect(kv).To(ContainSubstring("CBKVConnector"))
			Expect(kv).To(ContainSubstring(testSvcHost))

			By("M7: injection.imagePullSecrets appended to spec.imagePullSecrets")
			Expect(pullSecretNames(out)).To(ContainElement("cb-payload-pull"))

			By("M6: idempotency annotation stamped")
			Expect(out.Annotations[AnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Annotations).NotTo(HaveKey(AnnotationSkipReason))
		})
	})

	Describe("M7 image pull secrets", func() {
		It("does not duplicate a secret the pod already lists", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: "cb-payload-pull"}}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(pullSecretNames(out)).To(Equal([]string{"cb-payload-pull"}))
		})

		It("honors the cacheblend-image-pull-secrets annotation override", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Annotations[AnnotationImagePullSecrets] = "override-a, override-b"
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			names := pullSecretNames(out)
			Expect(names).To(ContainElements("override-a", "override-b"))
			Expect(names).NotTo(ContainElement("cb-payload-pull"))
		})
	})

	Describe("gating", func() {
		It("allows a pod with no engine annotation unchanged", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				delete(p.Annotations, AnnotationEngine)
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("skips + stamps engine-not-found when the connection ConfigMap is absent", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, false) // no connection ConfigMap seeded
			pod := vllmPod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[AnnotationSkipReason]).To(Equal(SkipReasonEngineNotFound))
			Expect(out.Annotations).NotTo(HaveKey(AnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
			Expect(out.Spec.InitContainers).To(BeEmpty())
		})

		It("skips + stamps engine-not-found when the engine CR is absent", func() {
			engine := newTestEngine(nil)
			// Seed an injector whose engine name differs from the pod's annotation.
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Annotations[AnnotationEngine] = "does-not-exist"
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[AnnotationSkipReason]).To(Equal(SkipReasonEngineNotFound))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})

		It("allows an already-injected pod as a no-op", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Annotations[AnnotationInjected] = valueTrue
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("skips + stamps command-override when the target container overrides command", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exec vllm serve"}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[AnnotationSkipReason]).To(Equal(SkipReasonCommandOverride))
			Expect(out.Annotations).NotTo(HaveKey(AnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
			Expect(out.Spec.InitContainers).To(BeEmpty())
		})
	})

	Describe("append-or-replace arg semantics", func() {
		It("replaces a pre-existing --attention-backend value", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Args = []string{"--attention-backend", "FLASH_ATTN", "--model", "m"}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, "vllm")

			Expect(argsHasFlagValue(c.Args, cbFlagAttentionBackend, cbValAttentionBackend)).To(BeTrue())
			Expect(argsHasFlagValue(c.Args, cbFlagAttentionBackend, "FLASH_ATTN")).To(BeFalse())
			// Not duplicated.
			Expect(countFlag(c.Args, cbFlagAttentionBackend)).To(Equal(1))
		})

		It("replaces a pre-existing --attention-backend=value (single-token form)", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Args = []string{"--attention-backend=FLASH_ATTN", "--model", "m"}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, "vllm")

			Expect(c.Args).To(ContainElement("--attention-backend=CUSTOM"))
			Expect(c.Args).NotTo(ContainElement("--attention-backend=FLASH_ATTN"))
		})

		It("skips + stamps when the user already supplies --kv-transfer-config", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Args = []string{"--kv-transfer-config", `{"kv_connector":"Other"}`}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, "vllm")

			By("the user's kv-transfer-config JSON is untouched")
			Expect(argsFlagValue(c.Args, cbFlagKVTransferConfig)).To(Equal(`{"kv_connector":"Other"}`))
			Expect(argsFlagValue(c.Args, cbFlagKVTransferConfig)).NotTo(ContainSubstring("CBKVConnector"))

			By("the skip reason is stamped but the rest of the injection still applies")
			Expect(out.Annotations[AnnotationSkipReason]).To(Equal(SkipReasonKVTransferConfigPresent))
			Expect(out.Annotations[AnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(argsHasFlagValue(c.Args, cbFlagBlockSize, cbValBlockSize)).To(BeTrue())
		})

		It("prepends to a pre-existing PYTHONPATH", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Env = []corev1.EnvVar{
					{Name: "PYTHONPATH", Value: "/opt/extra"},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, "vllm")

			Expect(envValue(c, pythonPathEnvName)).To(Equal("/cb-plugin:/opt/extra"))
		})
	})

	Describe("target container resolution", func() {
		It("injects into the annotation-named non-first container", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Annotations[AnnotationContainer] = "vllm"
				p.Spec.Containers = []corev1.Container{
					{Name: "sidecar", Image: "busybox", Args: []string{"sleep"}},
					{Name: "vllm", Image: "vllm/vllm-openai:latest", Args: []string{"--model", "m"}},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("the vLLM container is mutated")
			vllm := findContainer(out, "vllm")
			Expect(envValue(vllm, pythonPathEnvName)).To(Equal(cbPluginMountPath))
			Expect(argsHasFlagValue(vllm.Args, cbFlagAttentionBackend, cbValAttentionBackend)).To(BeTrue())

			By("the sidecar container is untouched")
			sidecar := findContainer(out, "sidecar")
			Expect(envValue(sidecar, pythonPathEnvName)).To(BeEmpty())
			Expect(sidecar.Args).To(Equal([]string{"sleep"}))
		})

		It("uses the engine injection.targetContainer default when set", func() {
			named := "vllm"
			engine := newTestEngine(func(e *lmcachev1alpha1.CacheBlendEngine) {
				e.Spec.Injection.TargetContainer = &named
			})
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Spec.Containers = []corev1.Container{
					{Name: "sidecar", Image: "busybox", Args: []string{"sleep"}},
					{Name: "vllm", Image: "vllm/vllm-openai:latest", Args: []string{"--model", "m"}},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			vllm := findContainer(out, "vllm")
			Expect(argsHasFlagValue(vllm.Args, cbFlagAttentionBackend, cbValAttentionBackend)).To(BeTrue())
			sidecar := findContainer(out, "sidecar")
			Expect(sidecar.Args).To(Equal([]string{"sleep"}))
		})

		It("skips + stamps target-container-not-found for an unknown container name", func() {
			engine := newTestEngine(nil)
			injector := newPodInjector(engine, true)
			pod := vllmPod(func(p *corev1.Pod) {
				p.Annotations[AnnotationContainer] = "does-not-exist"
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[AnnotationSkipReason]).To(Equal(SkipReasonTargetContainerNotFound))
			Expect(out.Annotations).NotTo(HaveKey(AnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
			Expect(out.Spec.InitContainers).To(BeEmpty())
			By("the original vLLM container is left untouched")
			vllm := findContainer(out, "vllm")
			Expect(envValue(vllm, pythonPathEnvName)).To(BeEmpty())
		})
	})

	Describe("cudagraph modes", func() {
		It("emits decode-only compilation config for full_decode_only", func() {
			mode := lmcachev1alpha1.CudagraphFullDecodeOnly
			engine := newTestEngine(func(e *lmcachev1alpha1.CacheBlendEngine) {
				e.Spec.Injection.Cudagraph = &mode
			})
			injector := newPodInjector(engine, true)
			pod := vllmPod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, "vllm")

			Expect(argsHasFlag(c.Args, cbFlagEnforceEager)).To(BeFalse())
			Expect(argsFlagValue(c.Args, cbFlagCompilationConfig)).To(ContainSubstring("FULL_DECODE_ONLY"))
		})
	})
})

// --- test arg helpers (mirror the package's two-token / =-token recognition) ---

// argsHasFlagValue reports whether args carries flag with the given value in
// either the two-token or single-token form.
func argsHasFlagValue(args []string, flag, value string) bool {
	return argsFlagValue(args, flag) == value
}

// argsFlagValue returns the value bound to flag in args (two-token or
// single-token form), or "" if the flag is absent.
func argsFlagValue(args []string, flag string) string {
	eqPrefix := flag + "="
	for i := range len(args) {
		if args[i] == flag && i+1 < len(args) {
			return args[i+1]
		}
		if after, ok := strings.CutPrefix(args[i], eqPrefix); ok {
			return after
		}
	}
	return ""
}

// countFlag returns how many times flag appears (two-token or single-token).
func countFlag(args []string, flag string) int {
	eqPrefix := flag + "="
	n := 0
	for _, a := range args {
		if a == flag || strings.HasPrefix(a, eqPrefix) {
			n++
		}
	}
	return n
}
