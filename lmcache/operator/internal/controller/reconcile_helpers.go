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
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// handleFinalizer migrates any LMCacheEngine carrying the legacy
// "lmcache.ai/cleanup" finalizer off of it. Owner references on every
// child resource (DaemonSet, Services, ConfigMap, managed Secret)
// already cause K8s garbage collection to cascade-delete them when
// the CR goes away, so no finalizer work is necessary today.
//
// Why this changed: a no-op finalizer blocks CR deletion whenever the
// controller pod isn't running (e.g. during cluster issues or a
// `kubectl delete -k config/default` that takes down the operator
// alongside the CRDs), with no upside since GC already does the work.
// See discussion in https://github.com/LMCache/LMCache/issues/2693.
//
// We may re-introduce a finalizer in the future for state K8s GC
// can't reach — e.g. evicting L2 Redis keys on CR delete, federation
// deregistration. When that happens, this function will grow real
// cleanup work alongside the legacy migration.
//
// Returns (err, done). If done is true the caller must return.
func (r *LMCacheEngineReconciler) handleFinalizer(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) (error, bool) {
	// On the deletion path we still need to clear the legacy finalizer
	// if a CR created by an older operator version is being deleted,
	// otherwise K8s would block on it forever.
	if engine.DeletionTimestamp != nil {
		if controllerutil.ContainsFinalizer(engine, finalizerName) {
			controllerutil.RemoveFinalizer(engine, finalizerName)
			if err := r.Update(ctx, engine); err != nil {
				return err, true
			}
		}
		return nil, true
	}

	// Pro-active migration on the create/update path: strip the
	// legacy finalizer from any CR that still has it from a prior
	// operator version, so future deletes don't deadlock.
	if controllerutil.ContainsFinalizer(engine, finalizerName) {
		controllerutil.RemoveFinalizer(engine, finalizerName)
		if err := r.Update(ctx, engine); err != nil {
			return err, true
		}
		// Return done=true so the controller requeues with a fresh
		// Get; continuing here would race with the resourceVersion
		// bump the Update just produced on the server.
		return nil, true
	}

	return nil, false
}

// validateAndSetCondition runs validation and updates the ConfigValid condition.
// Returns an error if validation fails (to stop reconciliation).
func (r *LMCacheEngineReconciler) validateAndSetCondition(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	errs := engine.ValidateSpec()

	if len(errs) > 0 {
		// Re-fetch to get the latest resourceVersion before status update.
		generation := engine.Generation
		if err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, engine); err != nil {
			return fmt.Errorf("failed to re-fetch engine for status update: %w", err)
		}
		meta.SetStatusCondition(&engine.Status.Conditions, metav1.Condition{
			Type:               lmcachev1alpha1.ConditionConfigValid,
			Status:             metav1.ConditionFalse,
			Reason:             "ValidationFailed",
			Message:            errs.ToAggregate().Error(),
			ObservedGeneration: generation,
		})
		engine.Status.Phase = lmcachev1alpha1.PhaseFailed
		engine.Status.ObservedGeneration = generation
		if err := r.Status().Update(ctx, engine); err != nil {
			return fmt.Errorf("failed to update status after validation failure: %w", err)
		}
		return fmt.Errorf("spec validation failed: %s", errs.ToAggregate().Error())
	}

	// ConfigValid=True condition is set in updateStatus (after re-fetch)
	// to avoid resourceVersion conflicts.
	return nil
}

// reconcileDaemonSet creates or updates the DaemonSet.
func (r *LMCacheEngineReconciler) reconcileDaemonSet(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	// Resolve the coordinator connection (ref -> Service URL) so the DaemonSet
	// builder emits --coordinator-url for server registration.
	conn, err := resources.ResolveCoordinatorConnection(ctx, r.Client, engine.Namespace, engine.Spec.Coordinator)
	if err != nil {
		return err
	}
	engine.Spec.Coordinator = conn

	desired := resources.BuildDaemonSet(engine)

	existing := &appsv1.DaemonSet{}
	err = r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	// Preserve immutable selector
	desired.Spec.Selector = existing.Spec.Selector
	desired.Spec.Template.Labels = resources.MergeLabels(
		existing.Spec.Selector.MatchLabels,
		desired.Spec.Template.Labels,
	)

	if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Template = desired.Spec.Template
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileLookupService creates or updates the node-local lookup Service.
func (r *LMCacheEngineReconciler) reconcileLookupService(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	desired := resources.BuildLookupService(engine)

	existing := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Ports = desired.Spec.Ports
	existing.Spec.InternalTrafficPolicy = desired.Spec.InternalTrafficPolicy
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileMetricsService creates or updates the headless metrics Service.
func (r *LMCacheEngineReconciler) reconcileMetricsService(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	desired := resources.BuildMetricsService(engine)

	existing := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Ports = desired.Spec.Ports
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileConnectionConfigMap creates or updates the connection ConfigMap.
func (r *LMCacheEngineReconciler) reconcileConnectionConfigMap(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	desired := resources.BuildConnectionConfigMap(engine)

	existing := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Data = desired.Data
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// reconcileServiceMonitor creates, updates, or deletes the ServiceMonitor.
func (r *LMCacheEngineReconciler) reconcileServiceMonitor(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	log := logf.FromContext(ctx)

	if !resources.ServiceMonitorEnabled(&engine.Spec) {
		// Delete ServiceMonitor if it exists
		existing := &monitoringv1.ServiceMonitor{}
		err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, existing)
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			// If the CRD is not installed, ignore the error
			if meta.IsNoMatchError(err) {
				return nil
			}
			return err
		}
		log.Info("Deleting ServiceMonitor", "name", engine.Name)
		return r.Delete(ctx, existing)
	}

	desired := resources.BuildServiceMonitor(engine)

	existing := &monitoringv1.ServiceMonitor{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
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

	if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec = desired.Spec
	existing.Labels = desired.Labels

	return r.Patch(ctx, existing, patch)
}

// updateStatus queries the DaemonSet and pods to compute status fields.
// It re-fetches the engine to get the latest resourceVersion, avoiding
// conflicts from watch events triggered by earlier reconcile steps.
func (r *LMCacheEngineReconciler) updateStatus(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	// Re-fetch to get the latest resourceVersion, avoiding conflicts
	// from watch events triggered by earlier reconcile steps (e.g.
	// DaemonSet/Service creation fires Owns watches).
	if err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, engine); err != nil {
		return err
	}

	// ConfigValid condition (set here after re-fetch so it's not lost).
	meta.SetStatusCondition(&engine.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionConfigValid,
		Status:             metav1.ConditionTrue,
		Reason:             "Valid",
		Message:            "Spec validation passed",
		ObservedGeneration: engine.Generation,
	})

	ds := &appsv1.DaemonSet{}
	err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, ds)
	if err != nil {
		if apierrors.IsNotFound(err) {
			engine.Status.Phase = lmcachev1alpha1.PhasePending
			engine.Status.DesiredInstances = 0
			engine.Status.ReadyInstances = 0
			engine.Status.ObservedGeneration = engine.Generation
			return r.Status().Update(ctx, engine)
		}
		return err
	}

	engine.Status.DesiredInstances = ds.Status.DesiredNumberScheduled
	engine.Status.ReadyInstances = ds.Status.NumberReady
	engine.Status.ObservedGeneration = engine.Generation

	// Compute phase
	switch {
	case ds.Status.DesiredNumberScheduled == 0:
		engine.Status.Phase = lmcachev1alpha1.PhasePending
	case ds.Status.NumberReady == ds.Status.DesiredNumberScheduled:
		engine.Status.Phase = lmcachev1alpha1.PhaseRunning
	case ds.Status.NumberReady > 0:
		engine.Status.Phase = lmcachev1alpha1.PhaseDegraded
	default:
		engine.Status.Phase = lmcachev1alpha1.PhasePending
	}

	// Set conditions
	meta.SetStatusCondition(&engine.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionAvailable,
		Status:             conditionBool(ds.Status.NumberReady > 0),
		Reason:             reasonFromReady(ds.Status.NumberReady > 0, "AtLeastOneReady", "NoReadyInstances"),
		Message:            fmt.Sprintf("%d/%d instances ready", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled),
		ObservedGeneration: engine.Generation,
	})

	allReady := ds.Status.NumberReady == ds.Status.DesiredNumberScheduled && ds.Status.DesiredNumberScheduled > 0
	meta.SetStatusCondition(&engine.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionAllInstancesReady,
		Status:             conditionBool(allReady),
		Reason:             reasonFromReady(allReady, "AllReady", "NotAllReady"),
		Message:            fmt.Sprintf("%d/%d instances ready", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled),
		ObservedGeneration: engine.Generation,
	})

	// Build endpoints from pods
	serverPort := int32(5555)
	if engine.Spec.Server != nil && engine.Spec.Server.Port != nil {
		serverPort = *engine.Spec.Server.Port
	}
	metricsPort := int32(9090)
	if engine.Spec.Prometheus != nil && engine.Spec.Prometheus.Port != nil {
		metricsPort = *engine.Spec.Prometheus.Port
	}

	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(engine.Namespace),
		client.MatchingLabels(resources.SelectorLabels(engine.Name)),
	); err != nil {
		return err
	}

	endpoints := make([]lmcachev1alpha1.EndpointStatus, 0, len(podList.Items))
	for i := range podList.Items {
		pod := &podList.Items[i]
		ready := false
		for _, cond := range pod.Status.Conditions {
			if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
				ready = true
				break
			}
		}
		endpoints = append(endpoints, lmcachev1alpha1.EndpointStatus{
			NodeName:    pod.Spec.NodeName,
			HostIP:      pod.Status.HostIP,
			PodName:     pod.Name,
			Port:        serverPort,
			MetricsPort: metricsPort,
			Ready:       ready,
		})
	}
	engine.Status.Endpoints = endpoints

	return r.Status().Update(ctx, engine)
}

func conditionBool(b bool) metav1.ConditionStatus {
	if b {
		return metav1.ConditionTrue
	}
	return metav1.ConditionFalse
}

func reasonFromReady(ready bool, trueReason, falseReason string) string {
	if ready {
		return trueReason
	}
	return falseReason
}

// reconcileRESPAuthSecret ensures a local copy of the RESP auth secret
// exists in the engine's namespace. If the source secret is in the same
// namespace, the DaemonSet references it directly via a managed copy.
// If cross-namespace, the operator reads the source and creates/updates
// a managed copy with ownerRef on the engine.
func (r *LMCacheEngineReconciler) reconcileRESPAuthSecret(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	log := logf.FromContext(ctx)
	spec := &engine.Spec

	// No RESP auth configured — clean up any stale managed secret.
	if spec.L2Backend == nil || spec.L2Backend.RESP == nil || spec.L2Backend.RESP.AuthSecretRef == nil {
		return r.deleteRESPAuthSecretIfExists(ctx, engine)
	}

	ref := spec.L2Backend.RESP.AuthSecretRef
	sourceNS := ref.Namespace
	if sourceNS == "" {
		sourceNS = engine.Namespace
	}
	localName := resources.RESPAuthSecretName(engine.Name)

	// Read the source secret.
	source := &corev1.Secret{}
	if err := r.Get(ctx, types.NamespacedName{Name: ref.Name, Namespace: sourceNS}, source); err != nil {
		return fmt.Errorf("failed to read RESP auth secret %s/%s: %w", sourceNS, ref.Name, err)
	}

	// Validate that the source secret contains the required "password" key.
	password, ok := source.Data["password"]
	if !ok || len(password) == 0 {
		return fmt.Errorf("RESP auth secret %s/%s is missing required 'password' key", sourceNS, ref.Name)
	}

	// Build the local managed copy.
	// Only "password" is required; "username" is optional (Redis Enterprise
	// often uses password-only auth).
	secretData := map[string][]byte{
		"password": password,
	}
	if u, ok := source.Data["username"]; ok {
		secretData["username"] = u
	}
	desired := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      localName,
			Namespace: engine.Namespace,
			Labels:    resources.StandardLabels(engine.Name),
		},
		Data: secretData,
	}

	existing := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{Name: localName, Namespace: engine.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(engine, desired, r.Scheme); err != nil {
				return err
			}
			log.Info("Creating managed RESP auth secret", "name", localName, "source", sourceNS+"/"+ref.Name)
			return r.Create(ctx, desired)
		}
		return err
	}

	// Update existing — apply ownerRef, data, and labels.
	if err := ctrl.SetControllerReference(engine, existing, r.Scheme); err != nil {
		return err
	}
	patch := client.MergeFrom(existing.DeepCopy())
	existing.Data = desired.Data
	existing.Labels = desired.Labels
	return r.Patch(ctx, existing, patch)
}

// deleteRESPAuthSecretIfExists removes the managed RESP auth secret
// if it exists (e.g. when authSecretRef is removed from the spec).
func (r *LMCacheEngineReconciler) deleteRESPAuthSecretIfExists(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) error {
	secret := &corev1.Secret{}
	name := resources.RESPAuthSecretName(engine.Name)
	err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: engine.Namespace}, secret)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	// Only delete if we own it.
	if metav1.IsControlledBy(secret, engine) {
		return r.Delete(ctx, secret)
	}
	return nil
}
