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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// CacheBlendEngineReconciler reconciles a CacheBlendEngine object. It mirrors
// LMCacheEngineReconciler but targets the blend_v3 engine workload and carries
// no finalizer of its own: a brand-new CRD has no legacy CRs to migrate, so
// owner-reference garbage collection alone cascade-deletes the child DaemonSet,
// Services, ConfigMap, and managed Secret when the CR is removed.
type CacheBlendEngineReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=cacheblendengines,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=cacheblendengines/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=cacheblendengines/finalizers,verbs=update

// Reconcile reconciles the CacheBlendEngine CR. It applies defaults, validates
// the spec, then converges the RESP auth Secret (only when RESP auth is
// configured), the blend_v3 DaemonSet, the node-local lookup Service, the
// headless metrics Service, the <engine>-connection ConfigMap, and the optional
// ServiceMonitor before updating status. Every child carries a controller
// reference for cascade deletion.
func (r *CacheBlendEngineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// 1. Fetch CR
	engine := &lmcachev1alpha1.CacheBlendEngine{}
	if err := r.Get(ctx, req.NamespacedName, engine); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// 2. Set defaults
	engine.SetDefaults()

	// 3. Validate
	if err := r.validateAndSetCondition(ctx, engine); err != nil {
		return ctrl.Result{}, err
	}

	// 4. Reconcile RESP auth secret (cross-namespace copy if needed)
	if err := r.reconcileRESPAuthSecret(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile RESP auth secret")
		return ctrl.Result{}, err
	}

	// 5. Reconcile DaemonSet
	if err := r.reconcileDaemonSet(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile DaemonSet")
		return ctrl.Result{}, err
	}

	// 6. Reconcile lookup Service (node-local discovery for vLLM)
	if err := r.reconcileLookupService(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile lookup Service")
		return ctrl.Result{}, err
	}

	// 7. Reconcile metrics Service
	if err := r.reconcileMetricsService(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile metrics Service")
		return ctrl.Result{}, err
	}

	// 8. Reconcile ConfigMap
	if err := r.reconcileConnectionConfigMap(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile ConfigMap")
		return ctrl.Result{}, err
	}

	// 9. Reconcile ServiceMonitor
	if err := r.reconcileServiceMonitor(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile ServiceMonitor")
		return ctrl.Result{}, err
	}

	// 10. Update status
	if err := r.updateStatus(ctx, engine); err != nil {
		log.Error(err, "Failed to update status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *CacheBlendEngineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&lmcachev1alpha1.CacheBlendEngine{}).
		Owns(&appsv1.DaemonSet{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.Secret{}).
		Named("cacheblendengine").
		Complete(r)
}
