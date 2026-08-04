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
	"fmt"
	"sync/atomic"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// nsSeq is a monotonic counter appended to every generated namespace
// name so concurrent specs (or repeated runs against the same envtest
// API server within a suite) never collide. envtest doesn't garbage-
// collect a deleted namespace's children, so per-spec namespaces
// double as a cheap form of test isolation.
var nsSeq atomic.Int64

// uniqueNS returns a fresh namespace name unique within this test
// process. The prefix makes it easy to spot the source of a stray
// object if a test crashes mid-flight.
func uniqueNS(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, nsSeq.Add(1))
}

// mustCreateNS creates a namespace and registers cleanup. Returns the
// name so callers can put resources in it. We don't rely on namespace
// teardown to clean the contents — envtest leaves them as Terminating
// without ever finalising the delete — so individual tests still must
// register cleanup for each resource they create.
func mustCreateNS(name string) string {
	GinkgoHelper()
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	DeferCleanup(func() {
		// Best-effort namespace delete. Don't fail the test on errors
		// — the suite tears down the whole apiserver after AfterSuite
		// anyway.
		_ = k8sClient.Delete(ctx, ns)
	})
	return name
}

// newEngine builds an LMCacheEngine fixture and Creates it in the
// cluster (which populates APIVersion / Kind / UID — required for
// ctrl.SetControllerReference to work in the function-under-test).
// Returns the Created object so callers can mutate Spec further.
func newEngine(namespace, name string) *lmcachev1alpha1.LMCacheEngine {
	GinkgoHelper()
	e := &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       lmcachev1alpha1.LMCacheEngineSpec{L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10}},
	}
	Expect(k8sClient.Create(ctx, e)).To(Succeed())
	DeferCleanup(func() { _ = k8sClient.Delete(ctx, e) })
	// Re-fetch so TypeMeta is populated for SetControllerReference.
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: namespace}, e)).To(Succeed())
	return e
}

// newReconciler wires a LMCacheEngineReconciler around the suite's
// envtest client. Kept as a one-liner because every spec needs one.
func newReconciler() *LMCacheEngineReconciler {
	return &LMCacheEngineReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
}

// ----- pure helpers (no envtest) -----------------------------------

var _ = Describe("conditionBool", func() {
	It("maps true → ConditionTrue and false → ConditionFalse", func() {
		Expect(conditionBool(true)).To(Equal(metav1.ConditionTrue))
		Expect(conditionBool(false)).To(Equal(metav1.ConditionFalse))
	})
})

var _ = Describe("reasonFromReady", func() {
	It("returns the true-reason when ready, the false-reason otherwise", func() {
		Expect(reasonFromReady(true, "Ready", "NotReady")).To(Equal("Ready"))
		Expect(reasonFromReady(false, "Ready", "NotReady")).To(Equal("NotReady"))
	})
})

// ----- reconcileRESPAuthSecret -------------------------------------

var _ = Describe("reconcileRESPAuthSecret", func() {
	var (
		r          *LMCacheEngineReconciler
		nsName     string
		engineName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("resp-auth"))
		engineName = "test-engine"
	})

	It("no-ops when authSecretRef is nil", func() {
		// Cover the early-return branch: L2Backend.RESP.AuthSecretRef
		// nil means there's nothing to mirror, AND the call should
		// also clean up any stale managed secret (delegated to
		// deleteRESPAuthSecretIfExists).
		engine := newEngine(nsName, engineName)
		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		// No managed secret should have been created.
		got := &corev1.Secret{}
		err := k8sClient.Get(ctx, types.NamespacedName{
			Name:      resources.RESPAuthSecretName(engineName),
			Namespace: nsName,
		}, got)
		Expect(apierrors.IsNotFound(err)).To(BeTrue(), "expected NotFound, got %v", err)
	})

	It("deletes a stale managed secret when authSecretRef is removed", func() {
		// Pre-create a managed secret owned by the engine, simulating
		// a previous reconcile that wired up auth. Then drop the spec
		// reference and confirm the helper cleans up. Use the
		// controllerutil helper to wire the OwnerReference via the
		// scheme — manually-built refs miss APIVersion/Kind because
		// envtest's client doesn't populate TypeMeta on Get.
		engine := newEngine(nsName, engineName)
		stale := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      resources.RESPAuthSecretName(engineName),
				Namespace: nsName,
			},
			Data: map[string][]byte{"password": []byte("old")},
		}
		Expect(controllerutil.SetControllerReference(engine, stale, k8sClient.Scheme())).To(Succeed())
		Expect(k8sClient.Create(ctx, stale)).To(Succeed())

		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		// Stale managed secret must be gone.
		err := k8sClient.Get(ctx, types.NamespacedName{Name: stale.Name, Namespace: nsName}, &corev1.Secret{})
		Expect(apierrors.IsNotFound(err)).To(BeTrue(), "expected stale managed secret to be deleted, got err=%v", err)
	})

	It("leaves a non-owned secret of the same name alone", func() {
		// Defensive: if an admin pre-created a Secret at the managed
		// name without an ownerRef, the operator must NOT delete it
		// — that's user data. deleteRESPAuthSecretIfExists checks
		// IsControlledBy and bails out on false.
		engine := newEngine(nsName, engineName)
		unrelated := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      resources.RESPAuthSecretName(engineName),
				Namespace: nsName,
				// No OwnerReferences: this Secret is not "owned" by anyone.
			},
			Data: map[string][]byte{"password": []byte("user-owned")},
		}
		Expect(k8sClient.Create(ctx, unrelated)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, unrelated) })

		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		// Still there.
		got := &corev1.Secret{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: unrelated.Name, Namespace: nsName}, got)).To(Succeed())
		Expect(got.Data["password"]).To(Equal([]byte("user-owned")))
	})

	It("creates a managed copy when the source secret exists (same-namespace)", func() {
		engine := newEngine(nsName, engineName)
		source := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "redis-auth", Namespace: nsName},
			Data: map[string][]byte{
				"username": []byte("admin"),
				"password": []byte("s3cret"), // pragma: allowlist secret
			},
		}
		Expect(k8sClient.Create(ctx, source)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, source) })

		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host:          "redis",
				Port:          6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{Name: "redis-auth"},
			},
		}
		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		got := &corev1.Secret{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      resources.RESPAuthSecretName(engineName),
			Namespace: nsName,
		}, got)).To(Succeed())
		Expect(got.Data).To(HaveKeyWithValue("password", []byte("s3cret")))
		Expect(got.Data).To(HaveKeyWithValue("username", []byte("admin")))
		Expect(metav1.IsControlledBy(got, engine)).To(BeTrue(), "managed secret must be owned by the engine")
	})

	It("creates a managed copy from a cross-namespace source", func() {
		// The whole point of this helper — pulling a Secret from
		// another namespace and mirroring it. Use a separate source
		// namespace to prove the cross-NS read path executes.
		sourceNS := mustCreateNS(uniqueNS("resp-auth-src"))
		engine := newEngine(nsName, engineName)
		source := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "shared-redis-auth", Namespace: sourceNS},
			Data:       map[string][]byte{"password": []byte("xns-pw")}, // pragma: allowlist secret
		}
		Expect(k8sClient.Create(ctx, source)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, source) })

		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis",
				Port: 6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{
					Name:      "shared-redis-auth",
					Namespace: sourceNS,
				},
			},
		}
		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		got := &corev1.Secret{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      resources.RESPAuthSecretName(engineName),
			Namespace: nsName, // mirror lands in the engine's namespace, not the source's
		}, got)).To(Succeed())
		Expect(got.Data).To(HaveKeyWithValue("password", []byte("xns-pw")))
		Expect(got.Data).NotTo(HaveKey("username"), "username should be absent when source omits it")
	})

	It("updates an existing managed copy when the source password changes", func() {
		// First reconcile establishes the mirror; second reconcile
		// after a source-side rotation must update the managed copy.
		// This exercises the Patch path (separate from the Create
		// path the previous spec covers).
		engine := newEngine(nsName, engineName)
		source := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "rotate-auth", Namespace: nsName},
			Data:       map[string][]byte{"password": []byte("v1")},
		}
		Expect(k8sClient.Create(ctx, source)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, source) })

		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis", Port: 6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{Name: "rotate-auth"},
			},
		}
		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		// Rotate the source.
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "rotate-auth", Namespace: nsName}, source)).To(Succeed())
		source.Data["password"] = []byte("v2")
		Expect(k8sClient.Update(ctx, source)).To(Succeed())

		// Re-reconcile should propagate.
		Expect(r.reconcileRESPAuthSecret(ctx, engine)).To(Succeed())

		got := &corev1.Secret{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      resources.RESPAuthSecretName(engineName),
			Namespace: nsName,
		}, got)).To(Succeed())
		Expect(got.Data).To(HaveKeyWithValue("password", []byte("v2")))
	})

	It("errors when the source secret is missing", func() {
		engine := newEngine(nsName, engineName)
		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis", Port: 6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{Name: "does-not-exist"},
			},
		}
		err := r.reconcileRESPAuthSecret(ctx, engine)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("does-not-exist"))
	})

	It("errors when the source secret is missing the password key", func() {
		// Defensive against operator mis-configuration: a Secret
		// pointed at by authSecretRef MUST have a "password" entry.
		// If it doesn't, surface a clear error rather than write a
		// broken managed copy.
		engine := newEngine(nsName, engineName)
		source := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "wrong-keys", Namespace: nsName},
			// "password" missing; only "user" provided.
			Data: map[string][]byte{"user": []byte("admin")},
		}
		Expect(k8sClient.Create(ctx, source)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, source) })

		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{
				Host: "redis", Port: 6379,
				AuthSecretRef: &lmcachev1alpha1.SecretReference{Name: "wrong-keys"},
			},
		}
		err := r.reconcileRESPAuthSecret(ctx, engine)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("password"))
	})
})

// ----- handleFinalizer ---------------------------------------------

var _ = Describe("handleFinalizer", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("finalizer"))
	})

	It("is a no-op on a fresh CR without the legacy finalizer", func() {
		engine := newEngine(nsName, "fresh")
		err, done := r.handleFinalizer(ctx, engine)
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeFalse(), "fresh CR should not short-circuit reconcile")
	})

	It("strips the legacy finalizer on the create/update path and requeues", func() {
		// Old-operator CRs may still carry the lmcache.ai/cleanup
		// finalizer. handleFinalizer must remove it so future
		// deletes don't deadlock. Per the source comment, the
		// function returns done=true after the strip so the caller
		// re-fetches with the new resourceVersion.
		engine := newEngine(nsName, "legacy")
		engine.Finalizers = []string{finalizerName}
		Expect(k8sClient.Update(ctx, engine)).To(Succeed())
		// Re-fetch to pick up the resourceVersion bump.
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())

		err, done := r.handleFinalizer(ctx, engine)
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue(), "expected the caller to short-circuit after the migration")

		// Confirm the finalizer is gone in the server's view.
		out := &lmcachev1alpha1.LMCacheEngine{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, out)).To(Succeed())
		Expect(out.Finalizers).NotTo(ContainElement(finalizerName))
	})

	It("clears the legacy finalizer on the deletion path", func() {
		// envtest's API server enforces the same "blocks deletion
		// while finalizers exist" semantics as a real cluster. So
		// Create with the legacy finalizer, Delete (which sets
		// DeletionTimestamp but does not actually remove the
		// object), then call handleFinalizer and verify the
		// finalizer is cleared so the API server can finish the GC.
		engine := newEngine(nsName, "deleting")
		engine.Finalizers = []string{finalizerName}
		Expect(k8sClient.Update(ctx, engine)).To(Succeed())
		Expect(k8sClient.Delete(ctx, engine)).To(Succeed())

		// Re-fetch — the object is still there (Terminating), with
		// DeletionTimestamp set, because the finalizer blocks GC.
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())
		Expect(engine.DeletionTimestamp).NotTo(BeNil())

		err, done := r.handleFinalizer(ctx, engine)
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue(), "on the deletion path the caller must short-circuit")

		// After the finalizer is removed the API server reaps the
		// object; the Get may either return NotFound or, if envtest
		// is mid-flight, the object with an empty finalizer list.
		// Either is acceptable.
		out := &lmcachev1alpha1.LMCacheEngine{}
		err = k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, out)
		if err == nil {
			Expect(out.Finalizers).NotTo(ContainElement(finalizerName))
		} else {
			Expect(apierrors.IsNotFound(err)).To(BeTrue())
		}
	})

	It("is a no-op on the deletion path when no legacy finalizer is present", func() {
		// Newer CRs that never had the legacy finalizer should still
		// short-circuit the rest of reconcile on the deletion path,
		// but without an Update call.
		engine := newEngine(nsName, "del-clean")
		Expect(k8sClient.Delete(ctx, engine)).To(Succeed())

		// On envtest without finalizers, the object may already be
		// gone after Delete. Re-fetch tolerantly.
		err := k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)
		if apierrors.IsNotFound(err) {
			Skip("envtest GC'd the object before we could run handleFinalizer; nothing to assert")
		}
		Expect(err).NotTo(HaveOccurred())
		Expect(engine.DeletionTimestamp).NotTo(BeNil())

		err, done := r.handleFinalizer(ctx, engine)
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
	})
})

// ----- validateAndSetCondition -------------------------------------

var _ = Describe("validateAndSetCondition", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("validate"))
	})

	It("returns nil and does NOT set ConfigValid for a valid spec", func() {
		// Per the source comment, on the happy path the helper does
		// NOT set ConfigValid=True itself — that happens later in
		// updateStatus to avoid a resourceVersion conflict. So a
		// valid spec leaves the condition list untouched here.
		engine := newEngine(nsName, "good")
		Expect(r.validateAndSetCondition(ctx, engine)).To(Succeed())

		got := &lmcachev1alpha1.LMCacheEngine{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		Expect(meta.FindStatusCondition(got.Status.Conditions, lmcachev1alpha1.ConditionConfigValid)).To(BeNil())
	})

	It("sets ConfigValid=False with ValidationFailed reason on an invalid spec", func() {
		// Build a CR that bypasses kubebuilder's CRD-level checks
		// (otherwise the API server would reject it at admission)
		// but trips ValidateSpec's L2-backend rule.
		engine := newEngine(nsName, "bad")
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())
		engine.Spec.L2Backend = &lmcachev1alpha1.L2BackendSpec{
			// Both RESP and Raw set — ValidateSpec rejects this as
			// "exactly one of resp or raw must be set, got multiple".
			RESP: &lmcachev1alpha1.RESPL2AdapterSpec{Host: "redis", Port: 6379},
			Raw:  &lmcachev1alpha1.RawL2AdapterSpec{Type: "mock"},
		}
		Expect(k8sClient.Update(ctx, engine)).To(Succeed())
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())

		err := r.validateAndSetCondition(ctx, engine)
		Expect(err).To(HaveOccurred(), "expected validateAndSetCondition to return the validation error")

		// Status condition must be set to False with the documented
		// reason; phase moves to Failed.
		got := &lmcachev1alpha1.LMCacheEngine{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		cond := meta.FindStatusCondition(got.Status.Conditions, lmcachev1alpha1.ConditionConfigValid)
		Expect(cond).NotTo(BeNil())
		Expect(cond.Status).To(Equal(metav1.ConditionFalse))
		Expect(cond.Reason).To(Equal("ValidationFailed"))
		Expect(got.Status.Phase).To(Equal(lmcachev1alpha1.PhaseFailed))
		Expect(got.Status.ObservedGeneration).To(Equal(got.Generation))
	})
})

// ----- reconcileServiceMonitor -------------------------------------

var _ = Describe("reconcileServiceMonitor", func() {
	// NOTE: envtest in this suite does NOT install the
	// monitoring.coreos.com ServiceMonitor CRD (CRDDirectoryPaths
	// only points at config/crd/bases, which is just the
	// LMCacheEngine CRD). So we exercise the two branches that
	// don't require the CRD: (a) ServiceMonitor disabled and
	// nothing to clean up, and (b) ServiceMonitor enabled but the
	// CRD is absent (the meta.IsNoMatchError branch). The
	// create / update / delete paths against a real ServiceMonitor
	// are covered by the e2e suite (S-3 / ServiceMonitor spec).
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("smon"))
	})

	It("no-ops when ServiceMonitor is disabled and nothing exists", func() {
		engine := newEngine(nsName, "no-mon")
		// Default spec has prometheus.serviceMonitor.enabled = nil
		// (i.e. disabled), so the helper enters the "delete if
		// exists" branch, hits NotFound or NoMatch, and returns nil.
		Expect(r.reconcileServiceMonitor(ctx, engine)).To(Succeed())
	})

	It("returns nil when ServiceMonitor is enabled but the CRD is absent", func() {
		// On clusters without prometheus-operator the create path
		// must short-circuit cleanly so the rest of reconcile
		// continues. The CRD-absent case surfaces as
		// meta.IsNoMatchError, which the helper swallows.
		engine := newEngine(nsName, "enabled-no-crd")
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())
		enabled := true
		engine.Spec.Prometheus = &lmcachev1alpha1.PrometheusSpec{
			Enabled: &enabled,
			ServiceMonitor: &lmcachev1alpha1.ServiceMonitorSpec{
				Enabled: &enabled,
			},
		}
		Expect(k8sClient.Update(ctx, engine)).To(Succeed())
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, engine)).To(Succeed())

		Expect(r.reconcileServiceMonitor(ctx, engine)).To(Succeed())
	})
})

// ----- resource-reconcile helpers (Create + Patch) -----------------
//
// All four functions (reconcileDaemonSet / reconcileLookupService /
// reconcileMetricsService / reconcileConnectionConfigMap) share the
// same shape:
//
//   1. Build desired via resources.BuildX(engine).
//   2. r.Get the existing — if NotFound, Create with ownerRef.
//   3. Otherwise Patch the subset of fields the helper owns.
//
// Each Describe below pairs a "create on first reconcile" spec with
// a "patch on second reconcile after a spec change" spec — that pair
// covers both branches and proves the diff actually propagates to
// the child resource (the helper isn't a no-op patch).

var _ = Describe("reconcileDaemonSet", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("ds"))
	})

	It("creates a DaemonSet on first reconcile", func() {
		engine := newEngine(nsName, "ds-create")
		Expect(r.reconcileDaemonSet(ctx, engine)).To(Succeed())

		got := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		Expect(metav1.IsControlledBy(got, engine)).To(BeTrue(), "DaemonSet must be owned by the engine")
		Expect(got.Spec.Template.Spec.Containers).To(HaveLen(1))
		// Default port flows through to the container args.
		Expect(got.Spec.Template.Spec.Containers[0].Args).To(ContainElements("--port", "5555"))
	})

	It("patches the DaemonSet when spec.server.port changes", func() {
		engine := newEngine(nsName, "ds-patch")
		Expect(r.reconcileDaemonSet(ctx, engine)).To(Succeed())

		// Mutate the local engine — reconcileDaemonSet works off the
		// passed-in object, no need to round-trip through Update.
		newPort := int32(6555)
		engine.Spec.Server = &lmcachev1alpha1.ServerSpec{Port: &newPort}
		Expect(r.reconcileDaemonSet(ctx, engine)).To(Succeed())

		got := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		Expect(got.Spec.Template.Spec.Containers[0].Args).To(ContainElements("--port", "6555"))
		// Existing object's selector is preserved across the patch
		// (DaemonSet selectors are immutable; reconcileDaemonSet
		// copies it onto desired before patching).
		Expect(got.Spec.Selector).NotTo(BeNil())
	})
})

var _ = Describe("reconcileLookupService", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("svc-lookup"))
	})

	It("creates the lookup Service on first reconcile", func() {
		engine := newEngine(nsName, "svc-create")
		Expect(r.reconcileLookupService(ctx, engine)).To(Succeed())

		got := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		Expect(metav1.IsControlledBy(got, engine)).To(BeTrue())
		Expect(got.Spec.InternalTrafficPolicy).NotTo(BeNil())
		Expect(*got.Spec.InternalTrafficPolicy).To(Equal(corev1.ServiceInternalTrafficPolicyLocal))
		// Default port is exposed under a port named "server".
		hasServerPort := false
		for _, p := range got.Spec.Ports {
			if p.Name == "server" && p.Port == 5555 {
				hasServerPort = true
			}
		}
		Expect(hasServerPort).To(BeTrue(), "expected a 'server' port at 5555, got %+v", got.Spec.Ports)
	})

	It("patches the lookup Service when spec.server.port changes", func() {
		engine := newEngine(nsName, "svc-patch")
		Expect(r.reconcileLookupService(ctx, engine)).To(Succeed())

		newPort := int32(7777)
		engine.Spec.Server = &lmcachev1alpha1.ServerSpec{Port: &newPort}
		Expect(r.reconcileLookupService(ctx, engine)).To(Succeed())

		got := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: nsName}, got)).To(Succeed())
		hasNewPort := false
		for _, p := range got.Spec.Ports {
			if p.Name == "server" && p.Port == newPort {
				hasNewPort = true
			}
		}
		Expect(hasNewPort).To(BeTrue(), "expected 'server' port to reflect the new value, got %+v", got.Spec.Ports)
	})
})

var _ = Describe("reconcileMetricsService", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("svc-metrics"))
	})

	It("creates the metrics Service on first reconcile", func() {
		engine := newEngine(nsName, "metrics-create")
		Expect(r.reconcileMetricsService(ctx, engine)).To(Succeed())

		// resources.BuildMetricsService names the service
		// "<engine>-metrics" (see internal/resources/service.go).
		got := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      engine.Name + "-metrics",
			Namespace: nsName,
		}, got)).To(Succeed())
		Expect(metav1.IsControlledBy(got, engine)).To(BeTrue())
		hasMetricsPort := false
		for _, p := range got.Spec.Ports {
			if p.Port == 9090 {
				hasMetricsPort = true
			}
		}
		Expect(hasMetricsPort).To(BeTrue(), "expected port 9090, got %+v", got.Spec.Ports)
	})

	It("patches the metrics Service when spec.prometheus.port changes", func() {
		engine := newEngine(nsName, "metrics-patch")
		Expect(r.reconcileMetricsService(ctx, engine)).To(Succeed())

		newPort := int32(9191)
		engine.Spec.Prometheus = &lmcachev1alpha1.PrometheusSpec{Port: &newPort}
		Expect(r.reconcileMetricsService(ctx, engine)).To(Succeed())

		got := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: engine.Name + "-metrics", Namespace: nsName}, got)).To(Succeed())
		hasNewPort := false
		for _, p := range got.Spec.Ports {
			if p.Port == newPort {
				hasNewPort = true
			}
		}
		Expect(hasNewPort).To(BeTrue(), "expected port %d, got %+v", newPort, got.Spec.Ports)
	})
})

var _ = Describe("reconcileConnectionConfigMap", func() {
	var (
		r      *LMCacheEngineReconciler
		nsName string
	)

	BeforeEach(func() {
		r = newReconciler()
		nsName = mustCreateNS(uniqueNS("cm"))
	})

	It("creates the connection ConfigMap on first reconcile", func() {
		engine := newEngine(nsName, "cm-create")
		Expect(r.reconcileConnectionConfigMap(ctx, engine)).To(Succeed())

		got := &corev1.ConfigMap{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      engine.Name + "-connection",
			Namespace: nsName,
		}, got)).To(Succeed())
		Expect(metav1.IsControlledBy(got, engine)).To(BeTrue())
		Expect(got.Data).To(HaveKey("kv-transfer-config.json"))
		// Default port (5555) shows up in the JSON blob.
		Expect(got.Data["kv-transfer-config.json"]).To(ContainSubstring("5555"))
	})

	It("patches the ConfigMap when spec.server.port changes", func() {
		engine := newEngine(nsName, "cm-patch")
		Expect(r.reconcileConnectionConfigMap(ctx, engine)).To(Succeed())

		newPort := int32(6555)
		engine.Spec.Server = &lmcachev1alpha1.ServerSpec{Port: &newPort}
		Expect(r.reconcileConnectionConfigMap(ctx, engine)).To(Succeed())

		got := &corev1.ConfigMap{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      engine.Name + "-connection",
			Namespace: nsName,
		}, got)).To(Succeed())
		Expect(got.Data["kv-transfer-config.json"]).To(ContainSubstring("6555"))
		Expect(got.Data["kv-transfer-config.json"]).NotTo(ContainSubstring("\"5555\""),
			"old port should be replaced, not appended")
	})
})
