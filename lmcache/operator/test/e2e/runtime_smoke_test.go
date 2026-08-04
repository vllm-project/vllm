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
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/LMCache/LMCache/test/utils"
)

// Runtime smoke (e2e_gpu): verifies that the values declared on the CR
// actually take effect inside the LMCache server process — not just on
// the K8s objects. The M1 (no-GPU) suite stops at the K8s artifact
// shape; this spec runs the pod and probes its HTTP frontend so a
// regression that, say, swallows --chunk-size in arg-parsing is caught
// even when the DaemonSet container args look correct.
//
// We probe `/status` because it returns is_healthy, chunk_size, and
// hash_algorithm in one payload — enough to catch arg-parsing
// regressions on those fields. The other CR fields (port, max_workers,
// http_port) remain covered by the M1 K8s-side specs that diff the
// DaemonSet container args.
var _ = Describe("LMCacheEngine runtime smoke (GPU)", Ordered, func() {
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
	})

	It("GET /status reflects the CR's chunk_size + hash_algorithm end-to-end", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_runtime.yaml", nsName, "")
		Expect(err).NotTo(HaveOccurred())

		// Capture the spec values we'll assert against /status below.
		// Reading via derefInt32-style accessors keeps the assertion
		// independent of the operator's defaulting logic — we test
		// "what the CR + defaults imply" vs "what the live server reports."
		expectedChunkSize := int32(128)
		expectedHashAlgorithm := "blake3"
		// Used for port-forward target; matches lmc_runtime.yaml.
		const httpPort int32 = 8080

		By("applying the runtime LMCacheEngine")
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())
		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("waiting for the LMCache DaemonSet pod to become Ready")
		// 8 min is intentional: this includes image pull on a cold node.
		// Container image is lmcache/vllm-openai:latest (~10GB).
		podName, err := utils.WaitDaemonSetPodReady(ctx, k8sClient, key, 8*time.Minute)
		Expect(err).NotTo(HaveOccurred(), "LMCache pod did not become Ready")

		By(fmt.Sprintf("port-forwarding to pod %s:%d", podName, httpPort))
		closer, baseURL, err := utils.PortForward(
			utils.PortForwardSpec{Namespace: nsName, Target: "pod/" + podName},
			fmt.Sprintf("0:%d", httpPort),
		)
		Expect(err).NotTo(HaveOccurred())
		defer closer()

		By("waiting for /healthcheck to return 200")
		Expect(utils.WaitHTTP200(ctx, baseURL+"/healthcheck", 2*time.Minute)).To(Succeed(),
			"LMCache HTTP frontend did not become healthy")

		By("fetching /status and asserting CR fields propagated to the live server")
		status := &statusPayload{}
		Expect(utils.HTTPGetJSON(ctx, baseURL+"/status", status)).To(Succeed())
		Expect(status.IsHealthy).To(BeTrue(),
			"status.is_healthy must be true — engine did not initialize cleanly")
		Expect(status.ChunkSize).To(Equal(expectedChunkSize),
			"status.chunk_size: live server disagrees with spec.server.chunkSize")
		Expect(status.HashAlgorithm).To(Equal(expectedHashAlgorithm),
			"status.hash_algorithm: live server disagrees with spec.server.hashAlgorithm")
	})
})

// statusPayload is the strict subset of /status that the runtime
// spec asserts against. Fields we don't assert on are intentionally
// omitted — the server adds operational fields (registered_gpu_ids,
// active_sessions, storage_manager) that change every run and would
// just be noise here.
type statusPayload struct {
	IsHealthy     bool   `json:"is_healthy"`
	ChunkSize     int32  `json:"chunk_size"`
	HashAlgorithm string `json:"hash_algorithm"`
}
