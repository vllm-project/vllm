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

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// coordinatorDefaultMetricsPort is the metrics port used when prometheus is
// enabled but no port is configured (mirrors the engine default).
const coordinatorDefaultMetricsPort int32 = 9090

// CoordinatorServiceName returns the name of the ClusterIP Service that fronts
// the coordinator. It equals the CR name so the in-cluster endpoint is the
// deterministic http://<name>.<namespace>.svc:<port>, which engines resolve from
// a coordinator reference.
func CoordinatorServiceName(name string) string {
	return name
}

// CoordinatorEndpoint returns the in-cluster base URL other components use to
// reach the coordinator.
func CoordinatorEndpoint(coordinator *lmcachev1alpha1.LMCacheCoordinator) string {
	return fmt.Sprintf(
		"http://%s.%s.svc:%d",
		CoordinatorServiceName(coordinator.Name),
		coordinator.Namespace,
		CoordinatorPort(&coordinator.Spec),
	)
}

// coordinatorPrometheusEnabled reports whether prometheus metrics are enabled
// for the coordinator. Unlike the engines, the coordinator defaults to disabled
// because it does not yet serve a /metrics endpoint.
func coordinatorPrometheusEnabled(spec *lmcachev1alpha1.LMCacheCoordinatorSpec) bool {
	if spec.Prometheus == nil {
		return false
	}
	return derefBool(spec.Prometheus.Enabled, false)
}

// coordinatorMetricsPort returns the configured metrics port (default 9090).
func coordinatorMetricsPort(spec *lmcachev1alpha1.LMCacheCoordinatorSpec) int32 {
	if spec.Prometheus == nil {
		return coordinatorDefaultMetricsPort
	}
	return derefInt32(spec.Prometheus.Port, coordinatorDefaultMetricsPort)
}

// CoordinatorServiceMonitorEnabled reports whether a ServiceMonitor should be
// created for the coordinator.
func CoordinatorServiceMonitorEnabled(coordinator *lmcachev1alpha1.LMCacheCoordinator) bool {
	spec := &coordinator.Spec
	if spec.Prometheus == nil || spec.Prometheus.ServiceMonitor == nil {
		return false
	}
	return derefBool(spec.Prometheus.ServiceMonitor.Enabled, false)
}

// BuildCoordinatorService creates the ClusterIP Service that exposes the
// coordinator's HTTP port for fleet-wide discovery.
func BuildCoordinatorService(coordinator *lmcachev1alpha1.LMCacheCoordinator) *corev1.Service {
	name := coordinator.Name
	port := CoordinatorPort(&coordinator.Spec)

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      CoordinatorServiceName(name),
			Namespace: coordinator.Namespace,
			Labels:    CoordinatorStandardLabels(name),
		},
		Spec: corev1.ServiceSpec{
			Selector: CoordinatorSelectorLabels(name),
			Ports: []corev1.ServicePort{
				{
					Name:     coordinatorPortName,
					Port:     port,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}

// BuildCoordinatorMetricsService creates a headless Service for Prometheus
// scraping of the coordinator. Callers must ensure prometheus is enabled.
func BuildCoordinatorMetricsService(coordinator *lmcachev1alpha1.LMCacheCoordinator) *corev1.Service {
	name := coordinator.Name

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-metrics", name),
			Namespace: coordinator.Namespace,
			Labels:    CoordinatorStandardLabels(name),
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  CoordinatorSelectorLabels(name),
			Ports: []corev1.ServicePort{
				{
					Name:     "metrics",
					Port:     coordinatorMetricsPort(&coordinator.Spec),
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}

// BuildCoordinatorServiceMonitor creates the ServiceMonitor CR for the
// coordinator. Callers must ensure CoordinatorServiceMonitorEnabled is true.
func BuildCoordinatorServiceMonitor(coordinator *lmcachev1alpha1.LMCacheCoordinator) *monitoringv1.ServiceMonitor {
	name := coordinator.Name
	smSpec := coordinator.Spec.Prometheus.ServiceMonitor

	interval := monitoringv1.Duration(derefString(smSpec.Interval, "30s"))
	labels := MergeLabels(CoordinatorStandardLabels(name), smSpec.Labels)

	return &monitoringv1.ServiceMonitor{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: coordinator.Namespace,
			Labels:    labels,
		},
		Spec: monitoringv1.ServiceMonitorSpec{
			Selector: metav1.LabelSelector{
				MatchLabels: CoordinatorSelectorLabels(name),
			},
			Endpoints: []monitoringv1.Endpoint{
				{
					Port:     "metrics",
					Interval: interval,
				},
			},
		},
	}
}
