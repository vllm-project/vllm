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

package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// KvTransferConfig is the typed schema of the kv-transfer-config.json
// document produced by the operator into the <name>-connection ConfigMap.
// Field tags mirror the JSON keys that vLLM's --kv-transfer-config expects.
type KvTransferConfig struct {
	KVConnector            string                 `json:"kv_connector"`
	KVConnectorModulePath  string                 `json:"kv_connector_module_path"`
	KVRole                 string                 `json:"kv_role"`
	KVConnectorExtraConfig KvConnectorExtraConfig `json:"kv_connector_extra_config"`
}

// KvConnectorExtraConfig holds the LMCache MP-mode connection string the
// vLLM connector uses to reach the LMCache server. The keys contain dots
// — vLLM treats them as flat strings, not nested paths.
type KvConnectorExtraConfig struct {
	Host string `json:"lmcache.mp.host"`
	Port string `json:"lmcache.mp.port"`
}

// ApplyLMC creates the LMCacheEngine if absent, or patches it (spec only)
// if it already exists. The status subresource is left untouched.
func ApplyLMC(ctx context.Context, c client.Client, lmc *lmcachev1alpha1.LMCacheEngine) error {
	existing := &lmcachev1alpha1.LMCacheEngine{}
	err := c.Get(ctx, client.ObjectKeyFromObject(lmc), existing)
	if apierrors.IsNotFound(err) {
		return c.Create(ctx, lmc)
	}
	if err != nil {
		return fmt.Errorf("get LMCacheEngine %s/%s: %w", lmc.Namespace, lmc.Name, err)
	}
	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec = lmc.Spec
	existing.Labels = lmc.Labels
	existing.Annotations = lmc.Annotations
	if err := c.Patch(ctx, existing, patch); err != nil {
		return fmt.Errorf("patch LMCacheEngine %s/%s: %w", lmc.Namespace, lmc.Name, err)
	}
	*lmc = *existing
	return nil
}

// WaitLMCReconciled polls until status.observedGeneration matches the
// CR's metadata.generation AND the ConfigValid condition is True. This
// is the readiness signal for the no-GPU smoke tier — pods do not need
// to be Running for the operator to have produced its K8s artifacts.
func WaitLMCReconciled(ctx context.Context, c client.Client, key types.NamespacedName, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		lmc := &lmcachev1alpha1.LMCacheEngine{}
		if err := c.Get(ctx, key, lmc); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		if lmc.Generation == 0 || lmc.Status.ObservedGeneration != lmc.Generation {
			return false, nil
		}
		cond := meta.FindStatusCondition(lmc.Status.Conditions, lmcachev1alpha1.ConditionConfigValid)
		return cond != nil && cond.Status == "True", nil
	})
}

// WaitLMCPhase polls until status.phase equals the requested phase value
// (e.g. "Running", "Pending"). Use WaitLMCReconciled instead when the
// observation contract is "controller saw the spec," not "pods are up."
func WaitLMCPhase(
	ctx context.Context,
	c client.Client,
	key types.NamespacedName,
	phase string,
	timeout time.Duration,
) error {
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		lmc := &lmcachev1alpha1.LMCacheEngine{}
		if err := c.Get(ctx, key, lmc); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		return lmc.Status.Phase == phase, nil
	})
}

// WaitLMCReady is a thin alias for WaitLMCPhase("Running"). It exists
// because the GPU tier (M3) reads more naturally as "wait until ready"
// — the no-GPU tier should not call this.
func WaitLMCReady(ctx context.Context, c client.Client, key types.NamespacedName, timeout time.Duration) error {
	return WaitLMCPhase(ctx, c, key, lmcachev1alpha1.PhaseRunning, timeout)
}

// GetConnectionConfig fetches the <name>-connection ConfigMap and parses
// kv-transfer-config.json into a typed struct. Returns an error if the
// ConfigMap is missing, the data key is missing, or the JSON is invalid.
func GetConnectionConfig(ctx context.Context, c client.Client, key types.NamespacedName) (*KvTransferConfig, error) {
	cm := &corev1.ConfigMap{}
	cmKey := types.NamespacedName{Namespace: key.Namespace, Name: key.Name + "-connection"}
	if err := c.Get(ctx, cmKey, cm); err != nil {
		return nil, fmt.Errorf("get ConfigMap %s: %w", cmKey, err)
	}
	raw, ok := cm.Data["kv-transfer-config.json"]
	if !ok {
		return nil, fmt.Errorf("ConfigMap %s missing key kv-transfer-config.json", cmKey)
	}
	cfg := &KvTransferConfig{}
	if err := json.Unmarshal([]byte(raw), cfg); err != nil {
		return nil, fmt.Errorf("decode kv-transfer-config.json: %w", err)
	}
	return cfg, nil
}

// PatchLMCSpec re-fetches the CR, applies mutate to the spec, and submits
// a merge patch. The mutate callback receives a pointer to the spec and
// must mutate in place. Use this for tests like the port-update spec
// that need to model "user kubectl-patches the spec."
func PatchLMCSpec(
	ctx context.Context,
	c client.Client,
	key types.NamespacedName,
	mutate func(*lmcachev1alpha1.LMCacheEngineSpec),
) error {
	lmc := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, key, lmc); err != nil {
		return fmt.Errorf("get LMCacheEngine %s: %w", key, err)
	}
	patch := client.MergeFrom(lmc.DeepCopy())
	mutate(&lmc.Spec)
	return c.Patch(ctx, lmc, patch)
}

// DeleteLMCAndWaitGC issues a foreground delete on the CR and waits until
// both the CR and its primary owned resources (DaemonSet, lookup Service,
// connection ConfigMap) are gone. Times out if a finalizer is stuck.
func DeleteLMCAndWaitGC(ctx context.Context, c client.Client, key types.NamespacedName, timeout time.Duration) error {
	lmc := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, key, lmc); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("get LMCacheEngine %s: %w", key, err)
	}
	if err := c.Delete(ctx, lmc); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("delete LMCacheEngine %s: %w", key, err)
	}

	cmKey := types.NamespacedName{Namespace: key.Namespace, Name: key.Name + "-connection"}
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		if exists, err := objectExists(ctx, c, key, &lmcachev1alpha1.LMCacheEngine{}); err != nil || exists {
			return false, err
		}
		if exists, err := objectExists(ctx, c, key, &appsv1.DaemonSet{}); err != nil || exists {
			return false, err
		}
		if exists, err := objectExists(ctx, c, key, &corev1.Service{}); err != nil || exists {
			return false, err
		}
		if exists, err := objectExists(ctx, c, cmKey, &corev1.ConfigMap{}); err != nil || exists {
			return false, err
		}
		return true, nil
	})
}

// objectExists returns true if the object at key still exists in the API.
// It treats NotFound as the only "absent" signal; any other error is
// surfaced so the caller can distinguish transient API failures.
func objectExists(ctx context.Context, c client.Client, key types.NamespacedName, obj client.Object) (bool, error) {
	err := c.Get(ctx, key, obj)
	if err == nil {
		return true, nil
	}
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	return false, err
}
