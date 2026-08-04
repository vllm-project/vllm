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
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"

	"github.com/LMCache/LMCache/test/utils"
)

// Field coverage smoke checks that non-default spec fields flow
// through to the reconciled K8s objects.
//
//   - ServiceMonitor:    enabling prometheus.serviceMonitor produces a
//     ServiceMonitor CR with the configured labels. Auto-skipped when
//     the Prometheus Operator CRDs are absent.
//   - extraArgs:         the user's --max-workers 4 wins over the
//     operator's auto-generated --max-workers 1 (extraArgs are
//     appended LAST by contract).
//   - resourceOverrides: an explicit resourceOverrides block fully
//     replaces the auto-computed memory request.
var _ = Describe("LMCacheEngine field coverage smoke (no-GPU)", Ordered, func() {
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

	It("creates a ServiceMonitor with the configured labels", func() {
		if !serviceMonitorCRDInstalled() {
			Skip("monitoring.coreos.com ServiceMonitor CRD not installed; skipping ServiceMonitor spec")
		}

		lmc, err := utils.NewLMCFromFixture("lmc_servicemonitor.yaml", nsName, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("eventually a ServiceMonitor exists with the configured labels")
		Eventually(func(g Gomega) {
			sm := &monitoringv1.ServiceMonitor{}
			g.Expect(k8sClient.Get(ctx, types.NamespacedName(key), sm)).To(Succeed())
			g.Expect(sm.Labels).To(HaveKeyWithValue("release", "kube-prometheus-stack"))
			g.Expect(sm.Spec.Endpoints).NotTo(BeEmpty())
			g.Expect(sm.Spec.Endpoints[0].Port).To(Equal("metrics"))
			g.Expect(string(sm.Spec.Endpoints[0].Interval)).To(Equal("30s"))
		}, 30*time.Second, time.Second).Should(Succeed())
	})

	It("lets spec.extraArgs override the auto-generated --max-workers", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "fields-extraargs")
		Expect(err).NotTo(HaveOccurred())
		lmc.Spec.ExtraArgs = []string{"--max-workers", "4"}
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("checking the DaemonSet container args show --max-workers=4 wins")
		Eventually(func(g Gomega) {
			ds := &appsv1.DaemonSet{}
			g.Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
			args := containerArgs(ds)
			// Both occurrences should be present — the auto-generated 1
			// and the override 4 — so we can prove the override is
			// strictly the LAST one rather than a clobber of the first.
			g.Expect(args).To(ContainElements("--max-workers", "1", "--max-workers", "4"))
			g.Expect(argValueLast(args, "--max-workers")).To(Equal("4"))
		}, 30*time.Second, time.Second).Should(Succeed())
	})

	It("lets spec.resourceOverrides replace the auto-computed memory", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "fields-resourceoverride")
		Expect(err).NotTo(HaveOccurred())
		lmc.Spec.ResourceOverrides = &corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("70Gi"),
			},
		}
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("checking the DaemonSet container resources reflect the override")
		Eventually(func(g Gomega) {
			ds := &appsv1.DaemonSet{}
			g.Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
			g.Expect(ds.Spec.Template.Spec.Containers).To(HaveLen(1))
			req := ds.Spec.Template.Spec.Containers[0].Resources.Requests
			g.Expect(req).To(HaveKey(corev1.ResourceMemory))
			memQty := req[corev1.ResourceMemory]
			g.Expect(memQty.String()).To(Equal("70Gi"))
		}, 30*time.Second, time.Second).Should(Succeed())
	})
})
