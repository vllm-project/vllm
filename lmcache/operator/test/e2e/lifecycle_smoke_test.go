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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

// Lifecycle smoke covers the three CR-edit scenarios:
//
//   - Update propagation: patching spec.server.port flows into the
//     ConfigMap data and DaemonSet container args.
//   - Finalizer cleanup:  deleting the CR removes every owned object
//     within 60s; anything longer signals a stuck finalizer (a real
//     bug we want to catch).
//   - Invalid spec:       l1.sizeGB=-1 is rejected by the API server
//     at admission time; the controller never sees it and no
//     DaemonSet/Service is created.
var _ = Describe("LMCacheEngine lifecycle smoke (no-GPU)", Ordered, func() {
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

	It("patches spec.server.port and the new value flows to ConfigMap and DaemonSet", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "lifecycle-update")
		Expect(err).NotTo(HaveOccurred())
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("baseline: ConfigMap and DaemonSet show the default port 5555")
		baseCfg, err := utils.GetConnectionConfig(ctx, k8sClient, key)
		Expect(err).NotTo(HaveOccurred())
		Expect(baseCfg.KVConnectorExtraConfig.Port).To(Equal("5555"))

		By("patching spec.server.port from 5555 to 6555")
		Expect(utils.PatchLMCSpec(ctx, k8sClient, types.NamespacedName(key),
			func(spec *lmcachev1alpha1.LMCacheEngineSpec) {
				if spec.Server == nil {
					spec.Server = &lmcachev1alpha1.ServerSpec{}
				}
				newPort := int32(6555)
				spec.Server.Port = &newPort
			},
		)).To(Succeed())

		By("eventually the ConfigMap reflects the new port")
		Eventually(func(g Gomega) {
			cfg, err := utils.GetConnectionConfig(ctx, k8sClient, key)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(cfg.KVConnectorExtraConfig.Port).To(Equal("6555"))
		}, 60*time.Second, time.Second).Should(Succeed())

		By("eventually the DaemonSet container args contain the new port")
		Eventually(func(g Gomega) {
			ds := &appsv1.DaemonSet{}
			g.Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
			g.Expect(argValue(containerArgs(ds), "--port")).To(Equal("6555"))
		}, 60*time.Second, time.Second).Should(Succeed())
	})

	It("deletes the CR and garbage-collects DaemonSet, Service, and ConfigMap within 60s", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "lifecycle-delete")
		Expect(err).NotTo(HaveOccurred())
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("ensuring the owned objects are present before deletion")
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), &appsv1.DaemonSet{})).To(Succeed())
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), &corev1.Service{})).To(Succeed())
		Expect(k8sClient.Get(ctx,
			types.NamespacedName{Namespace: key.Namespace, Name: key.Name + "-connection"},
			&corev1.ConfigMap{},
		)).To(Succeed())

		By("deleting the CR — finalizer + owner refs must clean up everything in 60s")
		Expect(utils.DeleteLMCAndWaitGC(ctx, k8sClient, types.NamespacedName(key), 60*time.Second)).To(Succeed())
	})

	It("rejects l1.sizeGB=-1 at admission and creates no owned objects", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "lifecycle-invalid")
		Expect(err).NotTo(HaveOccurred())
		lmc.Spec.L1.SizeGB = -1

		By("attempting to create the CR — API server must reject with 422 Invalid")
		applyErr := utils.ApplyLMC(ctx, k8sClient, lmc)
		Expect(applyErr).To(HaveOccurred(), "expected API server to reject sizeGB=-1")
		Expect(apierrors.IsInvalid(applyErr)).To(BeTrue(),
			"expected an Invalid (422) status error, got: %v", applyErr)

		key := engineKey(nsName, "lifecycle-invalid")

		By("ensuring no LMCacheEngine, DaemonSet, or Service was created")
		Expect(apierrors.IsNotFound(
			k8sClient.Get(ctx, types.NamespacedName(key), &lmcachev1alpha1.LMCacheEngine{}),
		)).To(BeTrue(), "rejected CR must not appear in the API")
		Expect(apierrors.IsNotFound(
			k8sClient.Get(ctx, types.NamespacedName(key), &appsv1.DaemonSet{}),
		)).To(BeTrue(), "no DaemonSet may exist for a rejected CR")
		Expect(apierrors.IsNotFound(
			k8sClient.Get(ctx, types.NamespacedName(key), &corev1.Service{}),
		)).To(BeTrue(), "no Service may exist for a rejected CR")
	})
})
