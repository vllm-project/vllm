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
	"encoding/json"
	"fmt"
	"strings"

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

// cbeResourceName is the name of the CacheBlendEngine fixture reconciled by the
// controller tests; the owner-reference helper checks against it.
const cbeResourceName = "test-cbe"

// argsContainFlagValue reports whether the ["--flag", "value", ...] slice
// contains the given two-token flag/value pair.
func argsContainFlagValue(args []string, flag, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

// ownedBy reports whether owners contains a controller reference to the
// CacheBlendEngine test fixture (cbeResourceName).
func ownedBy(owners []metav1.OwnerReference) bool {
	for _, o := range owners {
		if o.Kind == "CacheBlendEngine" && o.Name == cbeResourceName && o.Controller != nil && *o.Controller {
			return true
		}
	}
	return false
}

var _ = Describe("CacheBlendEngine Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = cbeResourceName

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}

		BeforeEach(func() {
			By("creating the custom resource for the Kind CacheBlendEngine")
			engine := &lmcachev1alpha1.CacheBlendEngine{}
			err := k8sClient.Get(ctx, typeNamespacedName, engine)
			if err != nil && errors.IsNotFound(err) {
				// injection.payloadImage.repository is required by ValidateSpec
				// (the webhook needs it to inject a valid init container).
				payloadRepo := "lmcache/cacheblend-plugin"
				resource := &lmcachev1alpha1.CacheBlendEngine{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: lmcachev1alpha1.CacheBlendEngineSpec{
						L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
						Injection: &lmcachev1alpha1.InjectionSpec{
							PayloadImage: &lmcachev1alpha1.ImageSpec{Repository: &payloadRepo},
						},
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			resource := &lmcachev1alpha1.CacheBlendEngine{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance CacheBlendEngine")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			// Drain child resources so a subsequent test starts clean (envtest
			// has no GC controller, so ownerRef cascade deletion does not run).
			// The lookup Service shares the engine's name (no suffix).
			_ = k8sClient.Delete(ctx, &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: resourceName, Namespace: "default"}})
			_ = k8sClient.Delete(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: resourceName, Namespace: "default"}})
			_ = k8sClient.Delete(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: resourceName + "-metrics", Namespace: "default"}})
			_ = k8sClient.Delete(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: resourceName + "-connection", Namespace: "default"}})
		})

		It("should reconcile to a blend_v3 DaemonSet, Services, and connection ConfigMap with ownerRefs", func() {
			controllerReconciler := &CacheBlendEngineReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			By("Reconciling the created resource")
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Verifying the blend_v3 DaemonSet")
			ds := &appsv1.DaemonSet{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, ds)).To(Succeed())
			Expect(ownedBy(ds.OwnerReferences)).To(BeTrue())

			podSpec := ds.Spec.Template.Spec
			Expect(podSpec.HostIPC).To(BeTrue())
			Expect(podSpec.Containers).To(HaveLen(1))
			engineContainer := podSpec.Containers[0]

			Expect(argsContainFlagValue(engineContainer.Args, "--engine-type", "blend")).To(BeTrue())
			Expect(argsContainFlagValue(engineContainer.Args, "--l1-align-bytes", "16777216")).To(BeTrue())

			By("Verifying there is no GPU resource claim")
			_, hasGPU := engineContainer.Resources.Limits["nvidia.com/gpu"]
			Expect(hasGPU).To(BeFalse())

			By("Verifying the lookup Service is node-local (named after the engine)")
			lookupSvc := &corev1.Service{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, lookupSvc)).To(Succeed())
			Expect(ownedBy(lookupSvc.OwnerReferences)).To(BeTrue())
			Expect(lookupSvc.Spec.InternalTrafficPolicy).NotTo(BeNil())
			Expect(*lookupSvc.Spec.InternalTrafficPolicy).To(Equal(corev1.ServiceInternalTrafficPolicyLocal))

			By("Verifying the headless metrics Service")
			metricsSvc := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: resourceName + "-metrics", Namespace: "default"}, metricsSvc)).To(Succeed())
			Expect(ownedBy(metricsSvc.OwnerReferences)).To(BeTrue())
			Expect(metricsSvc.Spec.ClusterIP).To(Equal(corev1.ClusterIPNone))

			By("Verifying the connection ConfigMap carries CBKVConnector JSON")
			cm := &corev1.ConfigMap{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: resourceName + "-connection", Namespace: "default"}, cm)).To(Succeed())
			Expect(ownedBy(cm.OwnerReferences)).To(BeTrue())
			jsonStr, ok := cm.Data["kv-transfer-config.json"]
			Expect(ok).To(BeTrue())
			Expect(strings.Contains(jsonStr, "CBKVConnector")).To(BeTrue())

			config := map[string]any{}
			Expect(json.Unmarshal([]byte(jsonStr), &config)).To(Succeed())
			Expect(config["kv_connector"]).To(Equal("CBKVConnector"))

			// design §7: the connector must dial the node-local Service over TCP
			// (the key correction vs the single-machine model card), and carry the
			// blend tunables.
			extra, ok := config["kv_connector_extra_config"].(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(extra["lmcache.mp.host"]).To(Equal(
				fmt.Sprintf("tcp://%s.default.svc.cluster.local", resourceName)))
			Expect(extra).To(HaveKey("lmcache.mp.port"))
			Expect(extra).To(HaveKey("cb.check_layer"))
			Expect(extra).To(HaveKey("cb.recomp_ratio"))
		})

		It("should converge status with ConfigValid=True after reconcile", func() {
			controllerReconciler := &CacheBlendEngineReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			updated := &lmcachev1alpha1.CacheBlendEngine{}
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

			By("Verifying the observed generation tracks the spec")
			Expect(updated.Status.ObservedGeneration).To(Equal(updated.Generation))
			// envtest has no kubelet, so no instances are scheduled: phase Pending.
			Expect(updated.Status.Phase).To(Equal(lmcachev1alpha1.PhasePending))
		})
	})
})
