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
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// NewCBEFromFixture loads a fixture YAML, decodes it as a CacheBlendEngine,
// and overrides metadata.name and metadata.namespace with the supplied
// values. It mirrors NewLMCFromFixture for the CacheBlend engine type so the
// GPU smoke tier can parameterise a fixture against a fresh namespace. Callers
// can mutate the returned object further (e.g. inject env-driven images and
// pull secrets) before applying.
func NewCBEFromFixture(fixtureName, namespace, name string) (*lmcachev1alpha1.CacheBlendEngine, error) {
	data, err := LoadFixture(fixtureName)
	if err != nil {
		return nil, err
	}
	cbe := &lmcachev1alpha1.CacheBlendEngine{}
	if err := yaml.Unmarshal(data, cbe); err != nil {
		return nil, fmt.Errorf("decode fixture %q: %w", fixtureName, err)
	}
	if name != "" {
		cbe.Name = name
	}
	if namespace != "" {
		cbe.Namespace = namespace
	}
	// Strip any server-side metadata that may have leaked through.
	cbe.ResourceVersion = ""
	cbe.UID = ""
	cbe.Generation = 0
	return cbe, nil
}

// ApplyCBE creates the CacheBlendEngine if absent, or patches it (spec only)
// if it already exists. The status subresource is left untouched. Mirrors
// ApplyLMC.
func ApplyCBE(ctx context.Context, c client.Client, cbe *lmcachev1alpha1.CacheBlendEngine) error {
	existing := &lmcachev1alpha1.CacheBlendEngine{}
	err := c.Get(ctx, client.ObjectKeyFromObject(cbe), existing)
	if apierrors.IsNotFound(err) {
		return c.Create(ctx, cbe)
	}
	if err != nil {
		return fmt.Errorf("get CacheBlendEngine %s/%s: %w", cbe.Namespace, cbe.Name, err)
	}
	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec = cbe.Spec
	existing.Labels = cbe.Labels
	existing.Annotations = cbe.Annotations
	if err := c.Patch(ctx, existing, patch); err != nil {
		return fmt.Errorf("patch CacheBlendEngine %s/%s: %w", cbe.Namespace, cbe.Name, err)
	}
	*cbe = *existing
	return nil
}

// WaitCBEReconciled polls until status.observedGeneration matches the CR's
// metadata.generation AND the ConfigValid condition is True. This is the
// "controller saw and validated the spec" signal — it does NOT require the
// engine pod to be Running (use WaitDaemonSetPodReady for that, keyed on the
// same namespaced name since the engine DaemonSet is named after the CR).
// The CacheBlendEngine controller reuses the LMCacheEngine condition
// constants, so ConditionConfigValid is the right key here.
func WaitCBEReconciled(ctx context.Context, c client.Client, key types.NamespacedName, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		cbe := &lmcachev1alpha1.CacheBlendEngine{}
		if err := c.Get(ctx, key, cbe); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		if cbe.Generation == 0 || cbe.Status.ObservedGeneration != cbe.Generation {
			return false, nil
		}
		cond := meta.FindStatusCondition(cbe.Status.Conditions, lmcachev1alpha1.ConditionConfigValid)
		return cond != nil && cond.Status == "True", nil
	})
}
