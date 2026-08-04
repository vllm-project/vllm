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

package resources

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// ResolveCoordinatorConnection returns a copy of conn with URL populated. When
// conn references an LMCacheCoordinator (Ref), it is looked up in the given
// namespace and its in-cluster Service endpoint is used. An explicit URL is
// returned unchanged. A nil conn returns nil. The resolved copy is passed to the
// resource builders so BuildContainerArgs can emit --coordinator-url without a
// cluster read.
func ResolveCoordinatorConnection(
	ctx context.Context,
	c client.Client,
	namespace string,
	conn *lmcachev1alpha1.CoordinatorConnectionSpec,
) (*lmcachev1alpha1.CoordinatorConnectionSpec, error) {
	if conn == nil {
		return nil, nil
	}

	resolved := *conn
	if conn.URL != nil && *conn.URL != "" {
		return &resolved, nil
	}
	if conn.Ref == nil || conn.Ref.Name == "" {
		// Validation already rejects this; return unchanged so the builder emits
		// no coordinator flags rather than a malformed URL.
		return &resolved, nil
	}

	coordinator := &lmcachev1alpha1.LMCacheCoordinator{}
	if err := c.Get(ctx, types.NamespacedName{Name: conn.Ref.Name, Namespace: namespace}, coordinator); err != nil {
		return nil, fmt.Errorf("failed to resolve coordinator ref %q: %w", conn.Ref.Name, err)
	}

	url := CoordinatorEndpoint(coordinator)
	resolved.URL = &url
	return &resolved, nil
}
