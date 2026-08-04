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

package controller

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// lmccResourceName is the name of the LMCacheCoordinator fixture reconciled by
// the controller tests.
const lmccResourceName = "test-coordinator"

// ownedByCoordinator reports whether owners contains a controller reference to
// the LMCacheCoordinator test fixture.
func ownedByCoordinator(owners []metav1.OwnerReference) bool {
	for _, o := range owners {
		if o.Kind == "LMCacheCoordinator" && o.Name == lmccResourceName && o.Controller != nil && *o.Controller {
			return true
		}
	}
	return false
}

var _ = Describe("LMCacheCoordinator Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = lmccResourceName

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}

		BeforeEach(func() {
			By("creating the custom resource for the Kind LMCacheCoordinator")
			coordinator := &lmcachev1alpha1.LMCacheCoordinator{}
			err := k8sClient.Get(ctx, typeNamespacedName, coordinator)
			if err != nil && errors.IsNotFound(err) {
				resource := &lmcachev1alpha1.LMCacheCoordinator{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: lmcachev1alpha1.LMCacheCoordinatorSpec{},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			resource := &lmcachev1alpha1.LMCacheCoordinator{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance LMCacheCoordinator")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			// envtest has no GC controller, so drain children manually.
			_ = k8sClient.Delete(ctx, &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: resourceName, Namespace: "default"}})
			_ = k8sClient.Delete(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: resourceName, Namespace: "default"}})
		})

		It("should reconcile to a Deployment and Service with ownerRefs and config flags", func() {
			controllerReconciler := &LMCacheCoordinatorReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			By("Reconciling the created resource")
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Verifying the Deployment")
			deploy := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, deploy)).To(Succeed())
			Expect(ownedByCoordinator(deploy.OwnerReferences)).To(BeTrue())

			podSpec := deploy.Spec.Template.Spec
			Expect(podSpec.Containers).To(HaveLen(1))
			container := podSpec.Containers[0]
			Expect(container.Command).To(ContainElement("coordinator"))
			Expect(argsContainFlagValue(container.Args, "--port", "9300")).To(BeTrue())
			// Blend knobs are omitted when unset so the coordinator image
			// applies its own defaults (and older images stay compatible).
			Expect(container.Args).NotTo(ContainElement("--blend-chunk-size"))
			Expect(container.Args).NotTo(ContainElement("--blend-probe-stride"))

			By("Verifying the probe targets /healthz")
			Expect(container.ReadinessProbe).NotTo(BeNil())
			Expect(container.ReadinessProbe.HTTPGet).NotTo(BeNil())
			Expect(container.ReadinessProbe.HTTPGet.Path).To(Equal("/healthz"))

			By("Verifying the ClusterIP Service exposes the coordinator port")
			svc := &corev1.Service{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, svc)).To(Succeed())
			Expect(ownedByCoordinator(svc.OwnerReferences)).To(BeTrue())
			Expect(svc.Spec.Ports).To(HaveLen(1))
			Expect(svc.Spec.Ports[0].Port).To(Equal(int32(9300)))
		})

		It("should converge status with ConfigValid=True and a resolved endpoint", func() {
			controllerReconciler := &LMCacheCoordinatorReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			updated := &lmcachev1alpha1.LMCacheCoordinator{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, updated)).To(Succeed())

			By("Verifying the ConfigValid condition is True")
			var found bool
			for _, cond := range updated.Status.Conditions {
				if cond.Type == lmcachev1alpha1.ConditionConfigValid {
					found = true
					Expect(cond.Status).To(Equal(metav1.ConditionTrue))
				}
			}
			Expect(found).To(BeTrue())

			By("Verifying status fields")
			Expect(updated.Status.ObservedGeneration).To(Equal(updated.Generation))
			Expect(updated.Status.Endpoint).To(Equal("http://test-coordinator.default.svc:9300"))
			// envtest has no kubelet, so no replicas are ready: phase Pending.
			Expect(updated.Status.Phase).To(Equal(lmcachev1alpha1.PhasePending))
		})

		It("should reject an invalid spec via status", func() {
			By("Setting an out-of-range evictionRatio")
			resource := &lmcachev1alpha1.LMCacheCoordinator{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, resource)).To(Succeed())
			bad := 2.0
			resource.Spec.EvictionRatio = &bad
			Expect(k8sClient.Update(ctx, resource)).To(Succeed())

			controllerReconciler := &LMCacheCoordinatorReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).To(HaveOccurred())

			updated := &lmcachev1alpha1.LMCacheCoordinator{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, updated)).To(Succeed())
			Expect(updated.Status.Phase).To(Equal(lmcachev1alpha1.PhaseFailed))
		})
	})
})
