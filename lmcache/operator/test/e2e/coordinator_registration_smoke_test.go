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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

// Coordinator registration (e2e_gpu): verifies the new LMCacheCoordinator CRD
// end-to-end. The operator stands up the coordinator Deployment + Service, and a
// GPU-scheduled LMCacheEngine whose spec sets coordinator.ref resolves to the
// coordinator's in-cluster URL, registers on startup, and shows up in the
// coordinator's /instances fleet listing.
//
// This needs a GPU node because the engine runs the real lmcache server (the
// coordinator client only registers once the server's event loop starts). The
// coordinator pod itself is CPU-only; it is pinned to the GPU node here purely so
// both pods share the already-pulled lmcache image and the test stays fast.
var _ = Describe("LMCacheCoordinator registration smoke (GPU)", Ordered, func() {
	var (
		ctx    context.Context
		nsName string
	)

	const coordinatorName = "smoke-coordinator"

	BeforeEach(func() {
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
	})

	It("registers a GPU engine with the coordinator via coordinator.ref", func() {
		By("applying the LMCacheCoordinator")
		gpuSelector := map[string]string{"nvidia.com/gpu.present": "true"}
		coordinator := &lmcachev1alpha1.LMCacheCoordinator{
			ObjectMeta: metav1.ObjectMeta{Name: coordinatorName, Namespace: nsName},
			Spec: lmcachev1alpha1.LMCacheCoordinatorSpec{
				// Co-locate on the GPU worker so the (large) lmcache image is
				// pulled once and shared with the engine pod.
				NodeSelector: gpuSelector,
			},
		}
		Expect(k8sClient.Create(ctx, coordinator)).To(Succeed())

		By("waiting for the coordinator Deployment to become Available")
		// 12 min absorbs the cold pull of the ~10GB lmcache image.
		Expect(utils.WaitDeploymentAvailable(
			ctx, k8sClient,
			types.NamespacedName{Namespace: nsName, Name: coordinatorName},
			12*time.Minute,
		)).To(Succeed(), "coordinator Deployment did not become Available")

		By("applying a GPU LMCacheEngine that references the coordinator")
		lmc, err := utils.NewLMCFromFixture("lmc_runtime.yaml", nsName, "coord-engine")
		Expect(err).NotTo(HaveOccurred())
		lmc.Spec.Coordinator = &lmcachev1alpha1.CoordinatorConnectionSpec{
			Ref:              &corev1.LocalObjectReference{Name: coordinatorName},
			L2EventReporting: boolPtr(true),
		}
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("verifying the operator resolved coordinator.ref into --coordinator-url")
		ds := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, key, ds)).To(Succeed())
		wantURL := fmt.Sprintf("http://%s.%s.svc:9300", coordinatorName, nsName)
		Expect(argValue(containerArgs(ds), "--coordinator-url")).To(Equal(wantURL))

		By("waiting for the engine DaemonSet pod to become Ready")
		_, err = utils.WaitDaemonSetPodReady(ctx, k8sClient, key, 10*time.Minute)
		Expect(err).NotTo(HaveOccurred(), "LMCache engine pod did not become Ready")

		By("port-forwarding to the coordinator Service")
		closer, baseURL, err := utils.PortForward(
			utils.PortForwardSpec{Namespace: nsName, Target: "service/" + coordinatorName},
			"0:9300",
		)
		Expect(err).NotTo(HaveOccurred())
		defer closer()
		Expect(utils.WaitHTTP200(ctx, baseURL+"/healthz", 60*time.Second)).To(Succeed())

		By("verifying the engine registered in the coordinator fleet")
		// The server registers on startup; allow a short window after Ready.
		Eventually(func(g Gomega) {
			var resp struct {
				Instances []struct {
					InstanceID string `json:"instance_id"`
					IP         string `json:"ip"`
				} `json:"instances"`
			}
			g.Expect(utils.HTTPGetJSON(ctx, baseURL+"/instances", &resp)).To(Succeed())
			g.Expect(resp.Instances).NotTo(BeEmpty(), "no instances registered with the coordinator")
		}, 90*time.Second, 3*time.Second).Should(Succeed())
	})
})

// boolPtr returns a pointer to b (local helper to avoid importing a ptr pkg).
func boolPtr(b bool) *bool { return &b }
