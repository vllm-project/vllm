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
	"os/exec"
	"regexp"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/yaml"

	"github.com/LMCache/LMCache/test/utils"
)

// vLLM + LMCache round-trip (e2e_gpu): verifies that an external vLLM
// pod, when pointed at the operator-managed LMCacheEngine via the
// <engine>-connection ConfigMap and run with vLLM's APC OFF, actually
// stores KV on request 1 and retrieves it on request 2.
//
// The assertion grep's the LMCache server's stdout for its own
// "Stored N tokens" / "Retrieved N tokens" log lines (server.py logs
// these at INFO level on every store and retrieve). This is more
// direct than scraping Prometheus counters — those are emitted by
// the vLLM-side connector and depend on OTel export cadence; the
// server log is written synchronously and is the same signal an
// on-call would look at when debugging cache behaviour by hand.
//
// Knobs:
//   - VLLM_MODEL  Hugging Face model id. Default: Qwen/Qwen2.5-0.5B
//     (small enough to load on a 16GB GPU, big enough that one chunk
//     covers ≥256 tokens of the test prompt). Only the model config
//     and tokenizer are downloaded; weights are dummy (see fixture).
//   - VLLM_IMAGE  container image. Default: lmcache/vllm-openai:latest
//     (same image the operator pins for the LMCache DaemonSet, which
//     means a fresh node only pulls once).
//   - SKIP_VLLM_INTEGRATION  when set to "true", skip this spec.
//     Useful when the cluster has GPUs but no internet egress to HF
//     even for config/tokenizer files.
var _ = Describe("vLLM + LMCacheEngine integration smoke (GPU)", Ordered, func() {
	var (
		ctx    context.Context
		nsName string
	)

	BeforeEach(func() {
		if os.Getenv("SKIP_VLLM_INTEGRATION") == "true" {
			Skip("SKIP_VLLM_INTEGRATION=true; skipping vLLM round-trip spec")
		}
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
		// vLLM Deployment is owned by the spec, not by the LMCacheEngine,
		// so namespace deletion (via DeferCleanup in createTestNamespace)
		// is what cleans it up. No explicit teardown needed here.
	})

	It("stores KV on first request and retrieves it on a second identical request", func() {
		model := envDefault("VLLM_MODEL", "Qwen/Qwen2.5-0.5B")
		vllmImage := envDefault("VLLM_IMAGE", "lmcache/vllm-openai:latest")

		By("applying the runtime LMCacheEngine")
		lmc, err := utils.NewLMCFromFixture("lmc_runtime.yaml", nsName, "vllm-integration")
		Expect(err).NotTo(HaveOccurred())
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("waiting for the LMCache DaemonSet pod to become Ready")
		lmcPodName, err := utils.WaitDaemonSetPodReady(ctx, k8sClient, key, 8*time.Minute)
		Expect(err).NotTo(HaveOccurred(), "LMCache pod did not become Ready")

		By("applying the vLLM Deployment, parameterised against this engine")
		vllmRaw, err := utils.LoadFixture("vllm_deployment.yaml")
		Expect(err).NotTo(HaveOccurred())
		vllmYAML := substituteVLLMPlaceholders(string(vllmRaw), nsName, lmc.Name, model, vllmImage)
		vllmDeploy := &appsv1.Deployment{}
		Expect(yaml.Unmarshal([]byte(vllmYAML), vllmDeploy)).To(Succeed())
		Expect(k8sClient.Create(ctx, vllmDeploy)).To(Succeed())

		By("waiting for the vLLM Deployment to become Available (model load + connector handshake)")
		// 15 min — covers cold-image pull (~10GB), HF model download,
		// and vLLM startup. The readinessProbe gates on /v1/models so
		// the deployment is only Available once vLLM has fully loaded.
		Expect(utils.WaitDeploymentAvailable(
			ctx, k8sClient,
			types.NamespacedName{Namespace: nsName, Name: vllmDeploy.Name},
			15*time.Minute,
		)).To(Succeed(), "vLLM did not become Available — check pod events / logs")

		By("port-forwarding to vLLM service for completion requests")
		vllmCloser, vllmBaseURL, err := utils.PortForward(
			utils.PortForwardSpec{Namespace: nsName, Target: "deployment/" + vllmDeploy.Name},
			"0:8000",
		)
		Expect(err).NotTo(HaveOccurred())
		defer vllmCloser()
		Expect(utils.WaitHTTP200(ctx, vllmBaseURL+"/v1/models", 60*time.Second)).To(Succeed())

		// Build a prompt long enough to cross at least one LMCache
		// chunk boundary (chunkSize=128 tokens for the runtime
		// fixture). The repeated paragraph generates well over a
		// thousand tokens so the round-trip stores at least 8 chunks.
		prompt := buildLongPrompt()

		By("sending the first /v1/completions request (cold — LMCache should STORE)")
		Expect(postCompletion(ctx, vllmBaseURL, model, prompt)).To(Succeed())

		// Server-side flushes are synchronous up to the log line, but
		// kubelet's log streaming buffers a beat behind. A couple of
		// seconds is enough to absorb that without padding the suite.
		By("waiting 2s for the LMCache server log to flush")
		time.Sleep(2 * time.Second)

		By("verifying LMCache logged a Stored line after the first request")
		Eventually(func() int {
			return countLMCacheStored(ctx, nsName, lmcPodName)
		}, 30*time.Second, 2*time.Second).Should(BeNumerically(">=", 1),
			"LMCache did not log any 'Stored N tokens' line after request 1")
		storedBefore := countLMCacheStored(ctx, nsName, lmcPodName)
		retrievedBefore := countLMCacheRetrieved(ctx, nsName, lmcPodName)
		Expect(retrievedBefore).To(Equal(0),
			"LMCache logged a 'Retrieved' before any cache hit was possible (got %d)", retrievedBefore)

		By("sending the second /v1/completions request (same prompt — LMCache should HIT)")
		Expect(postCompletion(ctx, vllmBaseURL, model, prompt)).To(Succeed())

		By("verifying LMCache logged a Retrieved line after the second request")
		// Eventually: log buffering between server.py and kubectl logs
		// usually settles in <2s, but the storage controller is async
		// in its accounting so we poll up to 30s. Hitting the floor of
		// "retrieved-after >= 1 AND stored-after unchanged" proves the
		// second request rode the cache rather than re-storing.
		Eventually(func(g Gomega) {
			retrievedAfter := countLMCacheRetrieved(ctx, nsName, lmcPodName)
			storedAfter := countLMCacheStored(ctx, nsName, lmcPodName)
			g.Expect(retrievedAfter).To(BeNumerically(">=", 1),
				"LMCache did not log any 'Retrieved' line after request 2 (got %d)",
				retrievedAfter)
			g.Expect(storedAfter).To(Equal(storedBefore),
				"LMCache logged additional 'Stored' lines on the repeat request "+
					"(before=%d, after=%d) — the cache hit didn't short-circuit the store path",
				storedBefore, storedAfter)
		}, 30*time.Second, 2*time.Second).Should(Succeed())

		_, _ = fmt.Fprintf(GinkgoWriter,
			"cache round-trip assertion satisfied (stored=%d, retrieved=%d)\n",
			countLMCacheStored(ctx, nsName, lmcPodName),
			countLMCacheRetrieved(ctx, nsName, lmcPodName),
		)
	})
})

// substituteVLLMPlaceholders renders the vllm Deployment fixture for a
// concrete run. Kept as a tiny function so the template stays readable
// and so we don't pull in a heavy templating dependency for four substitutions.
func substituteVLLMPlaceholders(yamlText, ns, engineName, model, image string) string {
	r := strings.NewReplacer(
		"__NAMESPACE__", ns,
		"__ENGINE_NAME__", engineName,
		"__MODEL__", model,
		"__VLLM_IMAGE__", image,
	)
	return r.Replace(yamlText)
}

// completionRequest is the minimal /v1/completions payload — we don't
// need streaming, sampling, or beam parameters since we're measuring
// cache behaviour, not generation quality.
type completionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

// postCompletion sends a single /v1/completions request and discards
// the result. Failures (non-2xx, network errors, JSON parse errors)
// propagate to the spec so the assertion failure shows the underlying
// HTTP context, not just "cache miss."
func postCompletion(ctx context.Context, baseURL, model, prompt string) error {
	body := completionRequest{
		Model:       model,
		Prompt:      prompt,
		MaxTokens:   32,
		Temperature: 0,
	}
	// vllm inference can be slow on first request (kernel cache cold),
	// so the request gets its own longer timeout than the default
	// httpJSONClient. We use HTTPPostJSON with a derived context.
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	return utils.HTTPPostJSON(reqCtx, baseURL+"/v1/completions", body, nil)
}

// lmcacheStoredLine matches the LMCache MP server's stdout INFO line
// emitted on every successful STORE, e.g.
//
//	[2026-05-13 23:21:02,989] LMCache INFO: Stored 1152 tokens in 0.003 seconds
//
// The leading bracketed timestamp + log-level prefix may be coloured
// with ANSI escapes in TTY mode; we don't bother stripping those
// because `kubectl logs` returns the raw line and the regex only
// requires the literal "Stored " substring.
var lmcacheStoredLine = regexp.MustCompile(`LMCache INFO:.*\bStored \d+ tokens\b`)

// lmcacheRetrievedLine mirrors lmcacheStoredLine for the RETRIEVE path,
// emitted from server.py:531 on every cache hit.
var lmcacheRetrievedLine = regexp.MustCompile(`LMCache INFO:.*\bRetrieved \d+ tokens\b`)

// countLMCacheStored runs `kubectl logs` on the LMCache MP server pod
// and returns the number of matching "Stored N tokens" lines seen so
// far. Returns 0 on any kubectl error (caller is expected to wrap the
// call in Eventually, which already retries on transient failures).
func countLMCacheStored(ctx context.Context, ns, podName string) int {
	return countLogLines(ctx, ns, podName, lmcacheStoredLine)
}

// countLMCacheRetrieved is the RETRIEVE counterpart of countLMCacheStored.
func countLMCacheRetrieved(ctx context.Context, ns, podName string) int {
	return countLogLines(ctx, ns, podName, lmcacheRetrievedLine)
}

// countLogLines is the implementation behind countLMCacheStored /
// countLMCacheRetrieved. We shell out to kubectl rather than building
// a typed Pods/log client because the rest of the suite already
// depends on kubectl being on PATH for port-forward, and adding a
// streaming-log client just for this would be heavier than the
// problem warrants.
func countLogLines(ctx context.Context, ns, podName string, pattern *regexp.Regexp) int {
	cmd := exec.CommandContext(ctx, "kubectl", "logs",
		"-n", ns, podName,
		"--tail=10000", // generous: a few k tokens of LMCache INFO output
	)
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	return len(pattern.FindAllIndex(out, -1))
}

// buildLongPrompt assembles ~1.2k tokens of prose so the prompt spans
// at least 8 chunks at chunkSize=128. We use a repeated history
// passage rather than random tokens so the prompt is human-readable
// in dump output (helps debugging when the test fails) and so vLLM's
// tokeniser segments it consistently across runs.
func buildLongPrompt() string {
	const para = "The history and significance of the Roman empire spans more than a thousand years " +
		"and profoundly shaped Western civilization. Its legal, architectural, linguistic, and " +
		"political legacies persist to this day, influencing modern governments, languages, art, " +
		"engineering, and law. The empire's trajectory from the founding of Rome through the " +
		"Republic, the transition to the Principate under Augustus, the Pax Romana, the crisis of " +
		"the third century, the Dominate under Diocletian, the adoption of Christianity under " +
		"Constantine, the splitting into Western and Eastern halves, and the eventual collapse of " +
		"the West is one of history's great narratives. Key figures include Julius Caesar, " +
		"Augustus, Marcus Aurelius, Diocletian, Constantine, Justinian, and many others. "
	var b strings.Builder
	for range 8 {
		b.WriteString(para)
	}
	b.WriteString("Tell me a long, detailed story about the rise, peak, and eventual fall of Rome, " +
		"naming important figures and events.")
	return b.String()
}
