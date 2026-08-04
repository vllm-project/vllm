//go:build e2e
// +build e2e

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

package e2e

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/yaml"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

// LMCacheEngine injection webhook smoke (M1, CPU-only): proves the mutating
// webhook wires an opted-in vLLM pod to an LMCacheEngine end-to-end in a real
// cluster — the piece the handler unit tests cannot cover (the webhook
// registration, the objectSelector patch, RBAC, and the live connection
// ConfigMap read).
//
// The mutation happens at admission, so the pod carries the stamp and injected
// args/env the moment it exists — no GPU, no model load, no pod readiness
// required. Because failurePolicy is Ignore (fail-open), a broken webhook would
// admit the pod UNMUTATED and silently degrade to a non-cached run; we defend
// against that by asserting the success stamp and the injected wiring directly
// off the pod object.
var _ = Describe("LMCacheEngine injection webhook smoke", func() {
	var (
		ctx    context.Context
		nsName string
	)

	BeforeEach(func() {
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
		// The LMCacheEngine and the vLLM Deployment are both in the test
		// namespace, cleaned up by the namespace's DeferCleanup. No teardown.
	})

	It("stamps the vLLM pod and injects --kv-transfer-config + hostIPC + PYTHONHASHSEED", func() {
		By("applying a minimal LMCacheEngine")
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "inject-smoke")
		Expect(err).NotTo(HaveOccurred())
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("waiting for the <engine>-connection ConfigMap (the webhook reads it at admission)")
		// The Deployment must be created AFTER the ConfigMap exists: admission is
		// one-shot at pod CREATE, so a pod admitted before the ConfigMap exists
		// would be permanently unstamped (the webhook would fail open).
		connName := fmt.Sprintf("%s-connection", lmc.Name)
		Eventually(func() error {
			return k8sClient.Get(ctx, engineKey(nsName, connName), &corev1.ConfigMap{})
		}, 30*time.Second, 2*time.Second).Should(Succeed(),
			"engine connection ConfigMap %q never appeared", connName)

		By("creating the opted-in vLLM Deployment")
		raw, err := utils.LoadFixture("vllm_lmcache_injection_deployment.yaml")
		Expect(err).NotTo(HaveOccurred())
		yamlText := strings.NewReplacer(
			"__NAMESPACE__", nsName,
			"__ENGINE_NAME__", lmc.Name,
		).Replace(string(raw))
		dep := &appsv1.Deployment{}
		Expect(yaml.Unmarshal([]byte(yamlText), dep)).To(Succeed())
		Expect(k8sClient.Create(ctx, dep)).To(Succeed())

		By("verifying the mutating webhook injected LMCache into the vLLM pod")
		Eventually(func(g Gomega) {
			pod := firstPodForDeployment(ctx, nsName, dep)
			g.Expect(pod).NotTo(BeNil(), "no vLLM pod created yet")
			g.Expect(pod.Annotations).To(HaveKeyWithValue("lmcache.ai/lmcache-injected", "true"),
				"webhook did not stamp lmcache-injected=true; injection did not fire "+
					"(skip-reason=%q)", pod.Annotations["lmcache.ai/lmcache-skip-reason"])
			g.Expect(pod.Spec.HostIPC).To(BeTrue(), "webhook did not inject hostIPC=true")

			args := vllmContainerArgs(pod)
			g.Expect(argValue(args, "--kv-transfer-config")).To(ContainSubstring("LMCacheMPConnector"),
				"injected --kv-transfer-config missing the LMCacheMPConnector JSON: %v", args)
			g.Expect(podEnvValue(pod, "vllm", "PYTHONHASHSEED")).To(Equal("0"),
				"webhook did not inject PYTHONHASHSEED=0")
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		_, _ = fmt.Fprintf(GinkgoWriter, "LMCache injection smoke satisfied (engine=%s)\n", lmc.Name)
	})

	It("stages the lmcache payload when injection.payloadImage is set", func() {
		// Fixed staging contract the webhook injects (kept in lockstep with
		// internal/webhook/lmcache_inject_builders.go; the e2e package cannot
		// import those internal constants, so assert the observable literals).
		const (
			payloadRepo  = "registry.example.com/lmcache/lmcache-payload"
			payloadTag   = "e2e"
			payloadRef   = payloadRepo + ":" + payloadTag
			pullSecret   = "lmcache-payload-pull"
			initName     = "lmcache-payload-stage"
			payloadVol   = "lmcache-payload"
			payloadMount = "/lmcache-payload"
			sharedDirEnv = "SHARED_DIR"
		)

		By("applying an LMCacheEngine with injection.payloadImage set")
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "inject-payload-smoke")
		Expect(err).NotTo(HaveOccurred())
		repo, tag, policy := payloadRepo, payloadTag, "IfNotPresent"
		lmc.Spec.Injection = &lmcachev1alpha1.LMCacheInjectionSpec{
			PayloadImage:     &lmcachev1alpha1.ImageSpec{Repository: &repo, Tag: &tag, PullPolicy: &policy},
			ImagePullSecrets: []corev1.LocalObjectReference{{Name: pullSecret}},
		}
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("waiting for the <engine>-connection ConfigMap (the webhook reads it at admission)")
		connName := fmt.Sprintf("%s-connection", lmc.Name)
		Eventually(func() error {
			return k8sClient.Get(ctx, engineKey(nsName, connName), &corev1.ConfigMap{})
		}, 30*time.Second, 2*time.Second).Should(Succeed(),
			"engine connection ConfigMap %q never appeared", connName)

		By("creating the opted-in vLLM Deployment")
		raw, err := utils.LoadFixture("vllm_lmcache_injection_deployment.yaml")
		Expect(err).NotTo(HaveOccurred())
		yamlText := strings.NewReplacer(
			"__NAMESPACE__", nsName,
			"__ENGINE_NAME__", lmc.Name,
		).Replace(string(raw))
		dep := &appsv1.Deployment{}
		Expect(yaml.Unmarshal([]byte(yamlText), dep)).To(Succeed())
		Expect(k8sClient.Create(ctx, dep)).To(Succeed())

		By("verifying the mutating webhook staged the payload onto the vLLM pod")
		// The mutation is at admission, so the pod carries the staging the moment
		// it exists — the init container never has to pull or run (a bogus payload
		// repo ImagePullBackOffs harmlessly; we assert the pod SPEC only).
		Eventually(func(g Gomega) {
			pod := firstPodForDeployment(ctx, nsName, dep)
			g.Expect(pod).NotTo(BeNil(), "no vLLM pod created yet")
			g.Expect(pod.Annotations).To(HaveKeyWithValue("lmcache.ai/lmcache-injected", "true"),
				"webhook did not stamp lmcache-injected=true (skip-reason=%q)",
				pod.Annotations["lmcache.ai/lmcache-skip-reason"])

			By("payload init container: image + SHARED_DIR (no command override)")
			init := podInitContainer(pod, initName)
			g.Expect(init).NotTo(BeNil(), "payload init container %q not injected", initName)
			g.Expect(init.Image).To(Equal(payloadRef))
			g.Expect(init.Command).To(BeEmpty(), "init container must run the payload image ENTRYPOINT, not a command override")
			g.Expect(containerEnvValue(init, sharedDirEnv)).To(Equal(payloadMount))

			By("shared emptyDir payload volume")
			g.Expect(podHasEmptyDirVolume(pod, payloadVol)).To(BeTrue(), "payload emptyDir volume %q missing", payloadVol)

			By("read-only payload mount + PYTHONPATH on the vLLM container")
			vllm := podContainer(pod, "vllm")
			g.Expect(vllm).NotTo(BeNil())
			mount := containerVolumeMount(vllm, payloadVol)
			g.Expect(mount).NotTo(BeNil(), "payload mount %q missing on vLLM container", payloadVol)
			g.Expect(mount.MountPath).To(Equal(payloadMount))
			g.Expect(mount.ReadOnly).To(BeTrue())
			g.Expect(containerEnvValue(vllm, "PYTHONPATH")).To(Equal(payloadMount),
				"webhook did not prepend PYTHONPATH=%s", payloadMount)

			By("payload pull secret merged onto the pod")
			g.Expect(pod.Spec.ImagePullSecrets).To(ContainElement(corev1.LocalObjectReference{Name: pullSecret}))

			By("connection wiring still applied alongside staging")
			g.Expect(pod.Spec.HostIPC).To(BeTrue())
			g.Expect(argValue(vllmContainerArgs(pod), "--kv-transfer-config")).To(ContainSubstring("LMCacheMPConnector"))
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		_, _ = fmt.Fprintf(GinkgoWriter, "LMCache payload staging smoke satisfied (engine=%s)\n", lmc.Name)
	})
})

// podEnvValue returns the value of env var name on the named container of the
// pod, or "" if the container or var is absent.
func podEnvValue(pod *corev1.Pod, container, name string) string {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name != container {
			continue
		}
		for _, e := range pod.Spec.Containers[i].Env {
			if e.Name == name {
				return e.Value
			}
		}
	}
	return ""
}
