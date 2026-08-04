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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// LMCacheCoordinatorSpec defines the desired state of LMCacheCoordinator. The
// fields mirror the coordinator's MPCoordinatorConfig (see
// lmcache/v1/mp_coordinator/config.py); the controller renders them into the
// matching `lmcache coordinator` CLI flags.
type LMCacheCoordinatorSpec struct {
	// image defines the container image to use. The coordinator runs the same
	// lmcache binary as the engines, so it shares the default image.
	// +optional
	Image *ImageSpec `json:"image,omitempty"`

	// imagePullSecrets is a list of references to secrets for pulling the image.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// replicas is the number of coordinator pods. The coordinator's in-memory
	// registry is per-process, so running more than one replica only makes sense
	// behind a shared durable backend; the default of 1 fits the common case.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`

	// host is the address the coordinator's HTTP server binds to.
	// +optional
	// +kubebuilder:default="0.0.0.0"
	Host *string `json:"host,omitempty"`

	// port is the port the coordinator's HTTP server binds to.
	// +optional
	// +kubebuilder:default=9300
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port *int32 `json:"port,omitempty"`

	// instanceTimeout is the seconds without a heartbeat after which an instance
	// is evicted. Must be > 0.
	// +optional
	// +kubebuilder:default=30
	InstanceTimeout *float64 `json:"instanceTimeout,omitempty"`

	// healthCheckInterval is the seconds between health-check sweeps. 0 disables
	// the loop.
	// +optional
	// +kubebuilder:default=10
	HealthCheckInterval *float64 `json:"healthCheckInterval,omitempty"`

	// evictionCheckInterval is the seconds between L2 eviction sweeps. 0 disables
	// the loop.
	// +optional
	// +kubebuilder:default=5
	EvictionCheckInterval *float64 `json:"evictionCheckInterval,omitempty"`

	// evictionRatio is the fraction of tracked keys (by count) to evict per
	// cycle, in [0.0, 1.0].
	// +optional
	// +kubebuilder:default=0.2
	EvictionRatio *float64 `json:"evictionRatio,omitempty"`

	// triggerWatermark is the usage fraction of the quota that fires eviction,
	// in (0.0, 1.0].
	// +optional
	// +kubebuilder:default=1
	TriggerWatermark *float64 `json:"triggerWatermark,omitempty"`

	// blendChunkSize is the tokens per chunk for the global CacheBlend directory
	// (the match unit). It MUST equal the LMCache chunk size the blend servers
	// use, so the coordinator chunks published/queried tokens the same way.
	// When unset the coordinator image applies its own default (256); the
	// operator only passes --blend-chunk-size when this is explicitly set, so it
	// stays compatible with images whose CLI predates the flag.
	// +optional
	// +kubebuilder:validation:Minimum=1
	BlendChunkSize *int32 `json:"blendChunkSize,omitempty"`

	// blendProbeStride is the number of positions between CacheBlend match
	// probes. With partial-fill reuse any offset is usable, so 1 (probe every
	// offset) gives full recall; raise it only to trade recall for coordinator
	// CPU. When unset the coordinator image applies its own default (1); the
	// operator only passes --blend-probe-stride when this is explicitly set, so
	// it stays compatible with images whose CLI predates the flag.
	// +optional
	// +kubebuilder:validation:Minimum=1
	BlendProbeStride *int32 `json:"blendProbeStride,omitempty"`

	// prometheus defines Prometheus monitoring configuration. The coordinator
	// process does not yet expose a /metrics endpoint, so the ServiceMonitor is
	// disabled by default; enabling it is only useful once metrics are added.
	// +optional
	Prometheus *PrometheusSpec `json:"prometheus,omitempty"`

	// resourceOverrides allows overriding the coordinator pod's resource
	// requirements.
	// +optional
	ResourceOverrides *corev1.ResourceRequirements `json:"resourceOverrides,omitempty"`

	// logLevel is the log level for the coordinator process.
	// +optional
	// +kubebuilder:default="INFO"
	// +kubebuilder:validation:Enum=DEBUG;INFO;WARNING;ERROR
	LogLevel *string `json:"logLevel,omitempty"`

	// nodeSelector constrains which nodes the coordinator runs on.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// affinity defines pod scheduling affinity rules.
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// tolerations defines pod tolerations.
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// env defines additional environment variables.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// volumes defines additional volumes.
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// volumeMounts defines additional volume mounts.
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`

	// podAnnotations are additional annotations added to the coordinator pod.
	// +optional
	PodAnnotations map[string]string `json:"podAnnotations,omitempty"`

	// podLabels are additional labels added to the coordinator pod.
	// +optional
	PodLabels map[string]string `json:"podLabels,omitempty"`

	// serviceAccountName is the name of the ServiceAccount to use.
	// +optional
	ServiceAccountName string `json:"serviceAccountName,omitempty"`

	// priorityClassName is the priority class for the coordinator pod.
	// +optional
	PriorityClassName string `json:"priorityClassName,omitempty"`

	// extraArgs are additional CLI flags appended to the coordinator command.
	// They are appended last and can override any auto-generated flag.
	// +optional
	ExtraArgs []string `json:"extraArgs,omitempty"`
}

// LMCacheCoordinatorStatus defines the observed state of LMCacheCoordinator.
type LMCacheCoordinatorStatus struct {
	// phase is the overall phase of the LMCacheCoordinator.
	// +optional
	Phase string `json:"phase,omitempty"`

	// observedGeneration is the most recent generation observed.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// replicas is the number of desired coordinator pods.
	// +optional
	Replicas int32 `json:"replicas,omitempty"`

	// readyReplicas is the number of ready coordinator pods.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// endpoint is the in-cluster URL other components use to reach the
	// coordinator, e.g. http://<name>.<namespace>.svc:9300.
	// +optional
	Endpoint string `json:"endpoint,omitempty"`

	// conditions represent the current state of the LMCacheCoordinator resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=lmcc
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Replicas",type=integer,JSONPath=`.status.replicas`
// +kubebuilder:printcolumn:name="Endpoint",type=string,JSONPath=`.status.endpoint`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// LMCacheCoordinator is the Schema for the lmcachecoordinators API.
type LMCacheCoordinator struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of LMCacheCoordinator.
	// +required
	Spec LMCacheCoordinatorSpec `json:"spec"`

	// status defines the observed state of LMCacheCoordinator.
	// +optional
	Status LMCacheCoordinatorStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// LMCacheCoordinatorList contains a list of LMCacheCoordinator.
type LMCacheCoordinatorList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []LMCacheCoordinator `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LMCacheCoordinator{}, &LMCacheCoordinatorList{})
}
