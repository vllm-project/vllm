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
	"fmt"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// validateAndSetCondition runs validation and updates the ConfigValid condition.
// Returns an error if validation fails (to stop reconciliation).
func (r *LMCacheCoordinatorReconciler) validateAndSetCondition(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	errs := coordinator.ValidateSpec()

	if len(errs) > 0 {
		generation := coordinator.Generation
		if err := r.Get(ctx, types.NamespacedName{Name: coordinator.Name, Namespace: coordinator.Namespace}, coordinator); err != nil {
			return fmt.Errorf("failed to re-fetch coordinator for status update: %w", err)
		}
		meta.SetStatusCondition(&coordinator.Status.Conditions, metav1.Condition{
			Type:               lmcachev1alpha1.ConditionConfigValid,
			Status:             metav1.ConditionFalse,
			Reason:             "ValidationFailed",
			Message:            errs.ToAggregate().Error(),
			ObservedGeneration: generation,
		})
		coordinator.Status.Phase = lmcachev1alpha1.PhaseFailed
		coordinator.Status.ObservedGeneration = generation
		if err := r.Status().Update(ctx, coordinator); err != nil {
			return fmt.Errorf("failed to update status after validation failure: %w", err)
		}
		return fmt.Errorf("spec validation failed: %s", errs.ToAggregate().Error())
	}

	return nil
}

// reconcileDeployment creates or updates the coordinator Deployment.
func (r *LMCacheCoordinatorReconciler) reconcileDeployment(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	desired := resources.BuildCoordinatorDeployment(coordinator)

	existing := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	// Preserve immutable selector.
	desired.Spec.Selector = existing.Spec.Selector

	if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Replicas = desired.Spec.Replicas
	existing.Spec.Template = desired.Spec.Template
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileCoordinatorService creates or updates the ClusterIP Service.
func (r *LMCacheCoordinatorReconciler) reconcileCoordinatorService(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	desired := resources.BuildCoordinatorService(coordinator)

	existing := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Ports = desired.Spec.Ports
	existing.Spec.Selector = desired.Spec.Selector
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileCoordinatorServiceMonitor converges the headless metrics Service and
// the ServiceMonitor when prometheus is enabled, and deletes them otherwise.
func (r *LMCacheCoordinatorReconciler) reconcileCoordinatorServiceMonitor(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	log := logf.FromContext(ctx)

	if !resources.CoordinatorServiceMonitorEnabled(coordinator) {
		if err := r.deleteCoordinatorMetricsServiceIfExists(ctx, coordinator); err != nil {
			return err
		}
		// Delete ServiceMonitor if it exists.
		existing := &monitoringv1.ServiceMonitor{}
		err := r.Get(ctx, types.NamespacedName{Name: coordinator.Name, Namespace: coordinator.Namespace}, existing)
		if err != nil {
			if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
				return nil
			}
			return err
		}
		log.Info("Deleting coordinator ServiceMonitor", "name", coordinator.Name)
		return r.Delete(ctx, existing)
	}

	// Metrics Service.
	desiredSvc := resources.BuildCoordinatorMetricsService(coordinator)
	existingSvc := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desiredSvc.Name, Namespace: desiredSvc.Namespace}, existingSvc)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(coordinator, desiredSvc, r.Scheme); err != nil {
				return err
			}
			if err := r.Create(ctx, desiredSvc); err != nil {
				return err
			}
		} else {
			return err
		}
	} else {
		if err := ctrl.SetControllerReference(coordinator, desiredSvc, r.Scheme); err != nil {
			return err
		}
		patch := client.MergeFrom(existingSvc.DeepCopy())
		existingSvc.Spec.Ports = desiredSvc.Spec.Ports
		existingSvc.Spec.Selector = desiredSvc.Spec.Selector
		existingSvc.Labels = desiredSvc.Labels
		if err := r.Patch(ctx, existingSvc, patch); err != nil {
			return err
		}
	}

	// ServiceMonitor.
	desired := resources.BuildCoordinatorServiceMonitor(coordinator)
	existing := &monitoringv1.ServiceMonitor{}
	err = r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		if meta.IsNoMatchError(err) {
			log.Info("ServiceMonitor CRD not available, skipping")
			return nil
		}
		return err
	}

	if err := ctrl.SetControllerReference(coordinator, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec = desired.Spec
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// deleteCoordinatorMetricsServiceIfExists removes the headless metrics Service
// if it exists and is owned by this coordinator (e.g. when prometheus is
// disabled after being enabled).
func (r *LMCacheCoordinatorReconciler) deleteCoordinatorMetricsServiceIfExists(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	svc := &corev1.Service{}
	name := fmt.Sprintf("%s-metrics", coordinator.Name)
	err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: coordinator.Namespace}, svc)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if metav1.IsControlledBy(svc, coordinator) {
		return r.Delete(ctx, svc)
	}
	return nil
}

// updateStatus queries the Deployment to compute status fields. It re-fetches
// the coordinator to get the latest resourceVersion, avoiding conflicts from
// watch events triggered by earlier reconcile steps.
func (r *LMCacheCoordinatorReconciler) updateStatus(ctx context.Context, coordinator *lmcachev1alpha1.LMCacheCoordinator) error {
	if err := r.Get(ctx, types.NamespacedName{Name: coordinator.Name, Namespace: coordinator.Namespace}, coordinator); err != nil {
		return err
	}

	meta.SetStatusCondition(&coordinator.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionConfigValid,
		Status:             metav1.ConditionTrue,
		Reason:             "Valid",
		Message:            "Spec validation passed",
		ObservedGeneration: coordinator.Generation,
	})

	coordinator.Status.Endpoint = resources.CoordinatorEndpoint(coordinator)
	coordinator.Status.ObservedGeneration = coordinator.Generation

	deploy := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: coordinator.Name, Namespace: coordinator.Namespace}, deploy)
	if err != nil {
		if apierrors.IsNotFound(err) {
			coordinator.Status.Phase = lmcachev1alpha1.PhasePending
			coordinator.Status.Replicas = 0
			coordinator.Status.ReadyReplicas = 0
			return r.Status().Update(ctx, coordinator)
		}
		return err
	}

	desired := int32(0)
	if deploy.Spec.Replicas != nil {
		desired = *deploy.Spec.Replicas
	}
	ready := deploy.Status.ReadyReplicas
	coordinator.Status.Replicas = desired
	coordinator.Status.ReadyReplicas = ready

	switch {
	case desired == 0:
		coordinator.Status.Phase = lmcachev1alpha1.PhasePending
	case ready == desired:
		coordinator.Status.Phase = lmcachev1alpha1.PhaseRunning
	case ready > 0:
		coordinator.Status.Phase = lmcachev1alpha1.PhaseDegraded
	default:
		coordinator.Status.Phase = lmcachev1alpha1.PhasePending
	}

	meta.SetStatusCondition(&coordinator.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionAvailable,
		Status:             conditionBool(ready > 0),
		Reason:             reasonFromReady(ready > 0, "AtLeastOneReady", "NoReadyReplicas"),
		Message:            fmt.Sprintf("%d/%d replicas ready", ready, desired),
		ObservedGeneration: coordinator.Generation,
	})

	allReady := ready == desired && desired > 0
	meta.SetStatusCondition(&coordinator.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionAllInstancesReady,
		Status:             conditionBool(allReady),
		Reason:             reasonFromReady(allReady, "AllReady", "NotAllReady"),
		Message:            fmt.Sprintf("%d/%d replicas ready", ready, desired),
		ObservedGeneration: coordinator.Generation,
	})

	return r.Status().Update(ctx, coordinator)
}
