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

// LMCacheCoordinatorReconciler reconciles a LMCacheCoordinator object. It mirrors
// the engine reconcilers but targets the fleet coordinator service: it converges
// a Deployment (not a DaemonSet) plus a ClusterIP Service, and carries no
// finalizer of its own since owner-reference garbage collection cascade-deletes
// the children when the CR is removed.
type LMCacheCoordinatorReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcachecoordinators,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcachecoordinators/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcachecoordinators/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete

// Reconcile reconciles the LMCacheCoordinator CR. It applies defaults, validates
// the spec, then converges the coordinator Deployment, the ClusterIP Service,
// and the optional ServiceMonitor before updating status. Every child carries a
// controller reference for cascade deletion.
func (r *LMCacheCoordinatorReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// 1. Fetch CR
	coordinator := &lmcachev1alpha1.LMCacheCoordinator{}
	if err := r.Get(ctx, req.NamespacedName, coordinator); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// 2. Set defaults
	coordinator.SetDefaults()

	// 3. Validate
	if err := r.validateAndSetCondition(ctx, coordinator); err != nil {
		return ctrl.Result{}, err
	}

	// 4. Reconcile Deployment
	if err := r.reconcileDeployment(ctx, coordinator); err != nil {
		log.Error(err, "Failed to reconcile Deployment")
		return ctrl.Result{}, err
	}

	// 5. Reconcile Service
	if err := r.reconcileCoordinatorService(ctx, coordinator); err != nil {
		log.Error(err, "Failed to reconcile Service")
		return ctrl.Result{}, err
	}

	// 6. Reconcile metrics Service + ServiceMonitor (gated on prometheus)
	if err := r.reconcileCoordinatorServiceMonitor(ctx, coordinator); err != nil {
		log.Error(err, "Failed to reconcile ServiceMonitor")
		return ctrl.Result{}, err
	}

	// 7. Update status
	if err := r.updateStatus(ctx, coordinator); err != nil {
		log.Error(err, "Failed to update status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *LMCacheCoordinatorReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&lmcachev1alpha1.LMCacheCoordinator{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Named("lmcachecoordinator").
		Complete(r)
}
