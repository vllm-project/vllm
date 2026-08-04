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
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// ServiceMonitorEnabled returns true if the ServiceMonitor should be created.
func ServiceMonitorEnabled(spec *lmcachev1alpha1.LMCacheEngineSpec) bool {
	if spec.Prometheus == nil || spec.Prometheus.ServiceMonitor == nil {
		return false
	}
	return derefBool(spec.Prometheus.ServiceMonitor.Enabled, false)
}

// BuildServiceMonitor creates a ServiceMonitor CR for Prometheus Operator integration.
func BuildServiceMonitor(engine *lmcachev1alpha1.LMCacheEngine) *monitoringv1.ServiceMonitor {
	return buildServiceMonitorCore(engine.Name, engine.Namespace, &engine.Spec)
}

// buildServiceMonitorCore is the name/namespace/spec-keyed core shared by the
// LMCacheEngine and CacheBlendEngine ServiceMonitor builders. Callers must
// ensure ServiceMonitorEnabled(spec) is true (spec.Prometheus.ServiceMonitor is
// dereferenced here).
func buildServiceMonitorCore(name, namespace string, spec *lmcachev1alpha1.LMCacheEngineSpec) *monitoringv1.ServiceMonitor {
	smSpec := spec.Prometheus.ServiceMonitor

	interval := monitoringv1.Duration(derefString(smSpec.Interval, "30s"))

	labels := MergeLabels(StandardLabels(name), smSpec.Labels)

	return &monitoringv1.ServiceMonitor{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: monitoringv1.ServiceMonitorSpec{
			Selector: metav1.LabelSelector{
				MatchLabels: SelectorLabels(name),
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
