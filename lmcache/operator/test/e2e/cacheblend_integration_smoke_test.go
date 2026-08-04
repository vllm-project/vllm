//go:build e2e && e2e_gpu
// +build e2e,e2e_gpu

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
	"os"
	"regexp"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/yaml"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

// vLLM + CacheBlendEngine round-trip (e2e_gpu): verifies the full CacheBlend
// operator integration on a GPU node.
//
// The flow exercises every moving part:
//  1. A CacheBlendEngine reconciles into a node-local `lmcache server
//     --engine-type blend` DaemonSet pod.
//  2. A vLLM Deployment opts in to CacheBlend injection (label + engine
//     annotation). The mutating webhook injects the CacheBlend vLLM flags
//     (--attention-backend CUSTOM, --kv-transfer-config <engine>, etc.), the
//     cb-plugin init container, hostIPC, and the private payload's pull secret.
//  3. vLLM loads the CUSTOM attention backend (asserted on the injected args
//     AND on vLLM's own startup banner), starts, and serves /v1/models.
//  4. A completion request drives a forward pass, which makes vLLM's connector
//     register its rope cache with the engine — the engine logs
//     "Registered CB rope state for instance N".
//  5. The completion returns HTTP 200.
//
// Because failurePolicy on the webhook is Ignore (fail-open), a missing webhook
// would admit the vLLM pod UNMUTATED and silently degrade this to a non-blend
// run. We defend against that by asserting the webhook's success stamp
// (lmcache.ai/cacheblend-injected=true) and the injected arg directly off the
// pod object — not just on runtime behaviour.
//
// Knobs:
//   - VLLM_MODEL                    HF model id. Default Qwen/Qwen2.5-0.5B.
//   - VLLM_IMAGE                    vLLM image. Default lmcache/vllm-openai:latest-nightly.
//   - CACHEBLEND_ENGINE_IMAGE       blend server image. Default lmcache/vllm-openai:latest-nightly.
//   - CACHEBLEND_PAYLOAD_IMAGE      PRIVATE plugin image. Default
//     tensormesh/cacheblend-plugin:latest-nightly.
//   - CACHEBLEND_REGISTRY_USER      Docker registry username for the payload image.
//   - CACHEBLEND_REGISTRY_TOKEN     Docker registry password / PAT (read-only pull).
//   - CACHEBLEND_REGISTRY_SERVER    Registry server. Default https://index.docker.io/v1/.
//   - CACHEBLEND_BACKEND_LOG_PATTERN  Regex proving the CUSTOM backend loaded.
//     Default `Using AttentionBackendEnum\.CUSTOM backend`.
//   - SKIP_CACHEBLEND_INTEGRATION   "true" skips this spec.
//
// When CACHEBLEND_REGISTRY_USER/TOKEN are unset the spec Skips (the private
// payload image cannot be pulled), so the suite stays green on clusters
// without registry credentials.
var _ = Describe("vLLM + CacheBlendEngine integration smoke (GPU)", Ordered, func() {
	const pullSecretName = "cacheblend-registry"

	var (
		ctx    context.Context
		nsName string
	)

	BeforeEach(func() {
		if os.Getenv("SKIP_CACHEBLEND_INTEGRATION") == "true" {
			Skip("SKIP_CACHEBLEND_INTEGRATION=true; skipping CacheBlend round-trip spec")
		}
		if os.Getenv("CACHEBLEND_REGISTRY_USER") == "" || os.Getenv("CACHEBLEND_REGISTRY_TOKEN") == "" {
			Skip("CACHEBLEND_REGISTRY_USER/CACHEBLEND_REGISTRY_TOKEN unset; " +
				"cannot pull the private cacheblend-plugin payload image")
		}
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
		// The vLLM Deployment, the CacheBlendEngine, and the pull Secret are
		// all in the test namespace, so namespace deletion (DeferCleanup in
		// createTestNamespace) cleans everything up. No explicit teardown.
	})

	It("injects CacheBlend, registers rope state on the engine, and serves a completion", func() {
		model := envDefault("VLLM_MODEL", "Qwen/Qwen2.5-0.5B")
		// nightly by default: the engine, vLLM, and the latest-nightly payload
		// plugin must sit in the same CacheBlend compatibility window.
		vllmImage := envDefault("VLLM_IMAGE", "lmcache/vllm-openai:latest-nightly")
		engineImage := envDefault("CACHEBLEND_ENGINE_IMAGE", "lmcache/vllm-openai:latest-nightly")
		payloadImage := envDefault("CACHEBLEND_PAYLOAD_IMAGE", "tensormesh/cacheblend-plugin:latest-nightly")
		backendLog := regexp.MustCompile(
			envDefault("CACHEBLEND_BACKEND_LOG_PATTERN", `Using AttentionBackendEnum\.CUSTOM backend`))

		By("creating the private-registry pull Secret from env credentials")
		Expect(utils.CreateDockerConfigJSONSecret(
			ctx, k8sClient, nsName, pullSecretName,
			envDefault("CACHEBLEND_REGISTRY_SERVER", "https://index.docker.io/v1/"),
			os.Getenv("CACHEBLEND_REGISTRY_USER"),
			os.Getenv("CACHEBLEND_REGISTRY_TOKEN"),
		)).To(Succeed())

		By("applying the CacheBlendEngine")
		cbe, err := utils.NewCBEFromFixture("cacheblendengine.yaml", nsName, "cb-integration")
		Expect(err).NotTo(HaveOccurred())
		setImageSpec(&cbe.Spec.Image, engineImage)
		// The fixture always carries an injection block, but guard against a
		// future fixture edit dropping it — a nil Injection here would panic
		// rather than fail the spec cleanly.
		if cbe.Spec.Injection == nil {
			cbe.Spec.Injection = &lmcachev1alpha1.InjectionSpec{}
		}
		setImageSpec(&cbe.Spec.Injection.PayloadImage, payloadImage)
		Expect(utils.ApplyCBE(ctx, k8sClient, cbe)).To(Succeed())

		key := engineKey(nsName, cbe.Name)
		Expect(utils.WaitCBEReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("waiting for the CacheBlend engine DaemonSet pod to become Ready")
		// 8 min covers the cold pull of the engine image on a fresh GPU node.
		enginePod, err := utils.WaitDaemonSetPodReady(ctx, k8sClient, key, 8*time.Minute)
		Expect(err).NotTo(HaveOccurred(), "CacheBlend engine pod did not become Ready")

		By("verifying the blend engine logged its ZMQ listening line")
		Eventually(func() int {
			return countLogLines(ctx, nsName, enginePod, cbServerListeningLine)
		}, 60*time.Second, 2*time.Second).Should(BeNumerically(">=", 1),
			"blend engine never logged 'LMCache ZMQ cache server is running' — server did not bind")

		By("applying the vLLM Deployment that opts into CacheBlend injection")
		vllmRaw, err := utils.LoadFixture("vllm_cacheblend_deployment.yaml")
		Expect(err).NotTo(HaveOccurred())
		vllmYAML := substituteVLLMPlaceholders(string(vllmRaw), nsName, cbe.Name, model, vllmImage)
		vllmDeploy := &appsv1.Deployment{}
		Expect(yaml.Unmarshal([]byte(vllmYAML), vllmDeploy)).To(Succeed())
		Expect(k8sClient.Create(ctx, vllmDeploy)).To(Succeed())

		By("verifying the mutating webhook injected CacheBlend into the vLLM pod")
		// The mutation happens at admission, so the pod carries the stamp and
		// the injected args as soon as it exists — no need to wait for Running.
		Eventually(func(g Gomega) {
			pod := firstPodForDeployment(ctx, nsName, vllmDeploy)
			g.Expect(pod).NotTo(BeNil(), "no vLLM pod created yet")
			g.Expect(pod.Annotations).To(HaveKeyWithValue("lmcache.ai/cacheblend-injected", "true"),
				"webhook did not stamp cacheblend-injected=true; injection did not fire "+
					"(skip-reason=%q)", pod.Annotations["lmcache.ai/cacheblend-skip-reason"])
			args := vllmContainerArgs(pod)
			g.Expect(argValue(args, "--attention-backend")).To(Equal("CUSTOM"),
				"injected args missing '--attention-backend CUSTOM': %v", args)
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		By("waiting for the vLLM Deployment to become Available (model load + connector handshake)")
		// 15 min — cold image pull (~10GB) + payload init container pull +
		// model config download + vLLM startup. /v1/models gates readiness.
		// Fails fast (~seconds) if the private payload pull is wedged
		// (bad/missing pull secret, wrong tag) rather than burning the timeout.
		Expect(utils.WaitDeploymentAvailableOrImagePullError(
			ctx, k8sClient,
			types.NamespacedName{Namespace: nsName, Name: vllmDeploy.Name},
			15*time.Minute,
		)).To(Succeed(), "vLLM did not become Available — check pod events / logs")

		vllmPod := firstPodForDeployment(ctx, nsName, vllmDeploy)
		Expect(vllmPod).NotTo(BeNil(), "vLLM pod disappeared after the Deployment became Available")

		By("verifying vLLM logged the CUSTOM attention backend banner")
		Eventually(func() int {
			return countLogLines(ctx, nsName, vllmPod.Name, backendLog)
		}, 60*time.Second, 2*time.Second).Should(BeNumerically(">=", 1),
			"vLLM never logged the CUSTOM attention backend banner (pattern %q) — "+
				"the injected --attention-backend CUSTOM did not take effect", backendLog.String())

		By("port-forwarding to the vLLM service for completion requests")
		vllmCloser, vllmBaseURL, err := utils.PortForward(
			utils.PortForwardSpec{Namespace: nsName, Target: "deployment/" + vllmDeploy.Name},
			"0:8000",
		)
		Expect(err).NotTo(HaveOccurred())
		defer vllmCloser()
		Expect(utils.WaitHTTP200(ctx, vllmBaseURL+"/v1/models", 60*time.Second)).To(Succeed())

		By("sending a /v1/completions request and asserting HTTP 200")
		Expect(postCompletion(ctx, vllmBaseURL, model, buildLongPrompt())).To(Succeed())

		By("verifying the engine logged 'Registered CB rope state for instance N'")
		// The forward pass drives vLLM's connector to send CB_REGISTER_ROPE_V3,
		// which the engine handles and logs. kubectl log streaming buffers a
		// beat behind, so poll.
		Eventually(func() int {
			return countLogLines(ctx, nsName, enginePod, cbRopeRegisteredLine)
		}, 60*time.Second, 2*time.Second).Should(BeNumerically(">=", 1),
			"engine never logged 'Registered CB rope state' — the CacheBlend rope "+
				"handshake from vLLM did not reach the engine")

		_, _ = fmt.Fprintf(GinkgoWriter,
			"CacheBlend round-trip satisfied (engine pod=%s, vLLM pod=%s)\n",
			enginePod, vllmPod.Name)
	})
})

// cbServerListeningLine matches the blend engine's "server is listening" INFO
// line (server.py), proving the lmcache server bound its ZMQ socket.
var cbServerListeningLine = regexp.MustCompile(`LMCache ZMQ cache server is running`)

// cbRopeRegisteredLine matches the engine-side log emitted by
// BlendV3.cb_register_rope on every CB_REGISTER_ROPE_V3 message
// (modules/blend_v3.py). Its presence proves the vLLM connector negotiated
// the CacheBlend rope handshake with the engine.
var cbRopeRegisteredLine = regexp.MustCompile(`Registered CB rope state for instance \d+`)

// setImageSpec rewrites an ImageSpec's repository+tag from a "repo:tag" image
// reference (the env-knob form), allocating the ImageSpec if nil. A reference
// with no tag defaults the tag to "latest". The split is on the LAST colon, so
// it does not handle a registry-host:port prefix on an untagged reference —
// the smoke knobs never use that form.
func setImageSpec(spec **lmcachev1alpha1.ImageSpec, ref string) {
	repo, tag := ref, "latest"
	if i := strings.LastIndex(ref, ":"); i >= 0 && !strings.Contains(ref[i+1:], "/") {
		repo, tag = ref[:i], ref[i+1:]
	}
	if *spec == nil {
		*spec = &lmcachev1alpha1.ImageSpec{}
	}
	(*spec).Repository = &repo
	(*spec).Tag = &tag
}

// firstPodForDeployment and vllmContainerArgs live in smoke_helpers_test.go so
// the CPU-only LMCache injection smoke (build tag e2e, no e2e_gpu) can share
// them.
