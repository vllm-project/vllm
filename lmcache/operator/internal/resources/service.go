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
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// BuildLookupService creates a ClusterIP Service with internalTrafficPolicy=Local
// for node-local service discovery. vLLM pods connect to this service and kube-proxy
// routes traffic only to the LMCache pod on the same node.
func BuildLookupService(engine *lmcachev1alpha1.LMCacheEngine) *corev1.Service {
	return buildLookupServiceCore(engine.Name, engine.Namespace, &engine.Spec)
}

// buildLookupServiceCore is the name/namespace/spec-keyed core shared by the
// LMCacheEngine and CacheBlendEngine lookup-Service builders. The node-local
// internalTrafficPolicy=Local routing guarantee is owned here so it cannot drift.
func buildLookupServiceCore(name, namespace string, spec *lmcachev1alpha1.LMCacheEngineSpec) *corev1.Service {
	serverPort := derefInt32(getServerPort(spec), 5555)
	httpPort := getHTTPPort(spec)
	localPolicy := corev1.ServiceInternalTrafficPolicyLocal

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LookupServiceName(name),
			Namespace: namespace,
			Labels:    StandardLabels(name),
		},
		Spec: corev1.ServiceSpec{
			Selector:              SelectorLabels(name),
			InternalTrafficPolicy: &localPolicy,
			Ports: []corev1.ServicePort{
				{
					Name:     serverPortName,
					Port:     serverPort,
					Protocol: corev1.ProtocolTCP,
				},
				{
					Name:     "http",
					Port:     httpPort,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}

// BuildMetricsService creates a headless Service for Prometheus scraping.
func BuildMetricsService(engine *lmcachev1alpha1.LMCacheEngine) *corev1.Service {
	return buildMetricsServiceCore(engine.Name, engine.Namespace, &engine.Spec)
}

// buildMetricsServiceCore is the name/namespace/spec-keyed core shared by the
// LMCacheEngine and CacheBlendEngine metrics-Service builders.
func buildMetricsServiceCore(name, namespace string, spec *lmcachev1alpha1.LMCacheEngineSpec) *corev1.Service {
	promPort := int32(9090)
	if spec.Prometheus != nil {
		promPort = derefInt32(spec.Prometheus.Port, 9090)
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-metrics", name),
			Namespace: namespace,
			Labels:    StandardLabels(name),
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  SelectorLabels(name),
			Ports: []corev1.ServicePort{
				{
					Name:     "metrics",
					Port:     promPort,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}
