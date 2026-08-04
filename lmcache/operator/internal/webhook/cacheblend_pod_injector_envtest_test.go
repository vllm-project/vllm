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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/LMCache/LMCache/internal/resources"
)

// These specs exercise the FULL admission pipeline: a pod CREATE goes to the
// envtest API server, which calls the registered CacheBlendPodInjector webhook over TLS,
// and we read the persisted (mutated) pod back. This validates the wiring the
// fake-client unit tests cannot: the generated webhook manifest path, the
// decoder, and PatchResponseFromRaw round-tripping through the API server.
var _ = Describe("CacheBlendPodInjector webhook (envtest)", Ordered, func() {
	BeforeAll(func() {
		By("creating the target namespace")
		ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}
		Expect(client.IgnoreAlreadyExists(k8sClient.Create(envtestCtx, ns))).To(Succeed())

		By("creating the CacheBlendEngine and its connection ConfigMap")
		engine := newTestEngine(nil)
		Expect(k8sClient.Create(envtestCtx, engine)).To(Succeed())
		Expect(k8sClient.Create(envtestCtx, resources.BuildCBConnectionConfigMap(engine))).To(Succeed())
	})

	It("injects the plugin, flags, hostIPC, and private-image pull secret into an annotated pod", func() {
		pod := vllmPod(func(p *corev1.Pod) {
			p.Name = "vllm-injected"
			if p.Labels == nil {
				p.Labels = map[string]string{}
			}
			p.Labels["lmcache.ai/cacheblend-inject"] = valueTrue
		})
		Expect(k8sClient.Create(envtestCtx, pod)).To(Succeed())

		got := &corev1.Pod{}
		Expect(k8sClient.Get(envtestCtx,
			types.NamespacedName{Name: "vllm-injected", Namespace: testNamespace}, got)).To(Succeed())

		By("the idempotency annotation is stamped")
		Expect(got.Annotations).To(HaveKeyWithValue(AnnotationInjected, valueTrue))

		By("M0: hostIPC is set on the pod")
		Expect(got.Spec.HostIPC).To(BeTrue())

		By("M1/M2: the cb-plugin emptyDir volume and payload init container are present")
		Expect(hasPluginVolume(got)).To(BeTrue())
		Expect(got.Spec.InitContainers).NotTo(BeEmpty())
		init := got.Spec.InitContainers[0]
		Expect(init.Image).To(Equal("registry.example.com/lmcache/cacheblend-payload:pinned"))

		By("M3/M4: the vLLM container has the readOnly mount and PYTHONPATH")
		c := findContainer(got, "vllm")
		Expect(c).NotTo(BeNil())
		Expect(hasReadOnlyMount(c, "cb-plugin")).To(BeTrue())
		Expect(envValue(c, "PYTHONPATH")).To(ContainSubstring("/cb-plugin"))

		By("M5: every required vLLM flag is present, with the node-local CBKVConnector config")
		// Form-agnostic: the handler may emit two-token (--flag value) or
		// =-token (--flag=value) forms; argsHasFlagValue handles both.
		Expect(argsHasFlagValue(c.Args, "--attention-backend", "CUSTOM")).To(BeTrue())
		Expect(argsHasFlagValue(c.Args, "--block-size", "64")).To(BeTrue())
		Expect(argsHasFlagValue(c.Args, "--pipeline-parallel-size", "1")).To(BeTrue())
		Expect(c.Args).To(ContainElement("--no-enable-chunked-prefill"))
		Expect(c.Args).To(ContainElement("--no-async-scheduling"))
		kv := argsFlagValue(c.Args, "--kv-transfer-config")
		Expect(kv).To(ContainSubstring("CBKVConnector"))
		Expect(kv).To(ContainSubstring("tcp://" + testEngineName + "." + testNamespace + ".svc"))

		By("M7: the private payload pull secret is appended to the pod")
		Expect(pullSecretNames(got)).To(ContainElement("cb-payload-pull"))
	})

	It("leaves a pod without the engine annotation untouched", func() {
		pod := vllmPod(func(p *corev1.Pod) {
			p.Name = "vllm-no-annotation"
			p.Annotations = nil
		})
		Expect(k8sClient.Create(envtestCtx, pod)).To(Succeed())

		got := &corev1.Pod{}
		Expect(k8sClient.Get(envtestCtx,
			types.NamespacedName{Name: "vllm-no-annotation", Namespace: testNamespace}, got)).To(Succeed())

		Expect(got.Annotations).NotTo(HaveKey(AnnotationInjected))
		Expect(got.Spec.HostIPC).To(BeFalse())
		Expect(hasPluginVolume(got)).To(BeFalse())
		Expect(got.Spec.InitContainers).To(BeEmpty())
	})

	It("skips a pod whose target container overrides command, stamping a skip reason", func() {
		pod := vllmPod(func(p *corev1.Pod) {
			p.Name = "vllm-wrapped"
			p.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exec vllm serve"}
		})
		Expect(k8sClient.Create(envtestCtx, pod)).To(Succeed())

		got := &corev1.Pod{}
		Expect(k8sClient.Get(envtestCtx,
			types.NamespacedName{Name: "vllm-wrapped", Namespace: testNamespace}, got)).To(Succeed())

		Expect(got.Annotations).To(HaveKeyWithValue(AnnotationSkipReason, SkipReasonCommandOverride))
		Expect(got.Annotations).NotTo(HaveKey(AnnotationInjected))
		Expect(hasPluginVolume(got)).To(BeFalse())
	})

	It("injects across namespaces via a namespace-qualified engine reference", func() {
		const consumerNS = "cb-consumer"
		By("creating the consumer namespace")
		Expect(client.IgnoreAlreadyExists(k8sClient.Create(envtestCtx,
			&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: consumerNS}}))).To(Succeed())

		// Pod in another namespace, namespace-qualified ref to the engine in
		// testNamespace. Cross-namespace binds need no engine-side config: the
		// webhook resolves the engine + connection ConfigMap from the engine's
		// namespace and injects.
		pod := vllmPod(func(p *corev1.Pod) {
			p.Name = "vllm-xns"
			p.Namespace = consumerNS
			p.Annotations[AnnotationEngine] = testNamespace + "/" + testEngineName
		})
		Expect(k8sClient.Create(envtestCtx, pod)).To(Succeed())

		got := &corev1.Pod{}
		Expect(k8sClient.Get(envtestCtx,
			types.NamespacedName{Name: "vllm-xns", Namespace: consumerNS}, got)).To(Succeed())

		By("the pod is injected using the engine + ConfigMap from the engine's namespace")
		Expect(got.Annotations).To(HaveKeyWithValue(AnnotationInjected, valueTrue))
		Expect(got.Spec.HostIPC).To(BeTrue())
		Expect(hasPluginVolume(got)).To(BeTrue())
		Expect(got.Spec.InitContainers).NotTo(BeEmpty())
		c := findContainer(got, "vllm")
		Expect(c).NotTo(BeNil())
		kv := argsFlagValue(c.Args, "--kv-transfer-config")
		Expect(kv).To(ContainSubstring("tcp://" + testEngineName + "." + testNamespace + ".svc"))
	})
})

func hasPluginVolume(pod *corev1.Pod) bool {
	for _, v := range pod.Spec.Volumes {
		if v.Name == "cb-plugin" {
			return true
		}
	}
	return false
}

func hasReadOnlyMount(c *corev1.Container, name string) bool {
	for _, m := range c.VolumeMounts {
		if m.Name == name && m.ReadOnly {
			return true
		}
	}
	return false
}
