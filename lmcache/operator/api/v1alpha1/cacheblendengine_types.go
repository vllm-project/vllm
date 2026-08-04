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

// CacheBlendChunkSize is the only chunk size CacheBlend supports. The blend
// matcher requires chunk_size == vLLM --block-size (64) * 4 == 256, so the
// CacheBlendEngine server is locked to this value (see design §4).
const CacheBlendChunkSize int32 = 256

// Cudagraph mode constants for CacheBlendEngine injection.
const (
	// CudagraphEager forces eager execution (--enforce-eager). Default.
	CudagraphEager = "eager"
	// CudagraphPiecewise enables piecewise CUDA graph capture.
	CudagraphPiecewise = "piecewise"
	// CudagraphFullDecodeOnly enables full CUDA graphs for decode only.
	CudagraphFullDecodeOnly = "full_decode_only"
)

// BlendSpec defines the CacheBlend tunables injected into the vLLM connect-config.
type BlendSpec struct {
	// checkLayer is the layer index used by CacheBlend to decide which tokens
	// to recompute. It is surfaced to the connector as
	// kv_connector_extra_config["cb.check_layer"].
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	CheckLayer *int32 `json:"checkLayer,omitempty"`

	// recompRatio is the fraction of tokens CacheBlend recomputes. It is
	// surfaced to the connector as kv_connector_extra_config["cb.recomp_ratio"]
	// and must be in (0, 1].
	// +optional
	// +kubebuilder:default=0.15
	RecompRatio *float64 `json:"recompRatio,omitempty"`
}

// InjectionSpec defines the defaults the mutating webhook reads when injecting
// the CacheBlend payload into vLLM pods bound to this engine (see design §7, §8).
type InjectionSpec struct {
	// payloadImage is the init-container image (repository/tag/pullPolicy, like
	// spec.image) that stages the lmcache-cacheblend vLLM plugin into a shared
	// emptyDir. It is a SEPARATE, usually PRIVATE image: set
	// payloadImage.repository to your cacheblend-plugin image — the repository
	// default inherited from ImageSpec is the engine image and is NOT a valid
	// payload. For private registries, imagePullSecrets must reference Secret(s)
	// that exist in the vLLM pod's namespace.
	// +optional
	PayloadImage *ImageSpec `json:"payloadImage,omitempty"`

	// imagePullSecrets are appended to the vLLM pod's spec.imagePullSecrets so
	// the PRIVATE payload init-container image can pull. The referenced
	// Secret(s) must already exist in the vLLM pod's namespace; the operator
	// does not copy them cross-namespace.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// targetContainer is the name of the vLLM container to inject into. Empty
	// (the default) selects the first container; a per-pod annotation may
	// override it.
	// +optional
	TargetContainer *string `json:"targetContainer,omitempty"`

	// cudagraph selects the CUDA graph mode injected into the vLLM args. "eager"
	// (default) maps to --enforce-eager; "full_decode_only" enables decode-only
	// graphs. Full graphs are never used.
	// +optional
	// +kubebuilder:default="eager"
	// +kubebuilder:validation:Enum=eager;piecewise;full_decode_only
	Cudagraph *string `json:"cudagraph,omitempty"`
}

// CacheBlendEngineSpec defines the desired state of CacheBlendEngine. It mirrors
// LMCacheEngineSpec (reusing its shared sub-structs) and adds the blend tunables
// and injection defaults specific to CacheBlend.
type CacheBlendEngineSpec struct {
	// gpuVendor selects the GPU vendor. "nvidia" (default) requires the NVIDIA
	// GPU Operator's "nvidia" RuntimeClass; "amd" runs on the default container
	// runtime.
	// +optional
	// +kubebuilder:default="nvidia"
	// +kubebuilder:validation:Enum=nvidia;amd
	GPUVendor *string `json:"gpuVendor,omitempty"`

	// image defines the container image to use for the blend_v3 engine. This
	// may be a PRIVATE image; use imagePullSecrets to pull it.
	// +optional
	Image *ImageSpec `json:"image,omitempty"`

	// imagePullSecrets is a list of references to secrets for pulling the
	// engine image.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// server defines server configuration. chunkSize defaults to 256 because
	// CacheBlend requires chunk_size == 256.
	// +optional
	Server *ServerSpec `json:"server,omitempty"`

	// l1 defines the L1 memory cache configuration.
	L1 L1BackendSpec `json:"l1"`

	// eviction defines the cache eviction configuration.
	// +optional
	Eviction *EvictionSpec `json:"eviction,omitempty"`

	// prometheus defines Prometheus monitoring configuration.
	// +optional
	Prometheus *PrometheusSpec `json:"prometheus,omitempty"`

	// l2Backend defines the L2 storage backend.
	// Currently only a single adapter is supported.
	// +optional
	L2Backend *L2BackendSpec `json:"l2Backend,omitempty"`

	// coordinator configures registration with an MP coordinator. When unset,
	// the server does not register with any coordinator.
	// +optional
	Coordinator *CoordinatorConnectionSpec `json:"coordinator,omitempty"`

	// blend defines the CacheBlend tunables injected into the vLLM connect-config.
	// +optional
	Blend *BlendSpec `json:"blend,omitempty"`

	// injection defines the defaults the mutating webhook reads for pods bound
	// to this engine.
	// +optional
	Injection *InjectionSpec `json:"injection,omitempty"`

	// resourceOverrides allows overriding auto-computed resource requirements.
	// +optional
	ResourceOverrides *corev1.ResourceRequirements `json:"resourceOverrides,omitempty"`

	// logLevel is the log level for the LMCache server.
	// +optional
	// +kubebuilder:default="INFO"
	// +kubebuilder:validation:Enum=DEBUG;INFO;WARNING;ERROR
	LogLevel *string `json:"logLevel,omitempty"`

	// nodeSelector determines which nodes get a CacheBlend engine instance.
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

	// podAnnotations are additional annotations added to pods.
	// +optional
	PodAnnotations map[string]string `json:"podAnnotations,omitempty"`

	// podLabels are additional labels added to pods.
	// +optional
	PodLabels map[string]string `json:"podLabels,omitempty"`

	// serviceAccountName is the name of the ServiceAccount to use.
	// +optional
	ServiceAccountName string `json:"serviceAccountName,omitempty"`

	// priorityClassName is the priority class for the pods.
	// +optional
	PriorityClassName string `json:"priorityClassName,omitempty"`

	// privileged runs the engine container in privileged mode. On some clusters
	// this is required for the engine to see all node GPUs (for CUDA IPC) without
	// claiming any via the nvidia.com/gpu device plugin; on many clusters
	// NVIDIA_VISIBLE_DEVICES=all already grants that visibility without it, so it
	// defaults to false. Set it to true only on clusters where the engine cannot
	// otherwise see the GPUs.
	// +optional
	// +kubebuilder:default=false
	Privileged *bool `json:"privileged,omitempty"`

	// extraArgs are additional CLI flags appended to the server command.
	// They are appended last and can override any auto-generated flag.
	// +optional
	ExtraArgs []string `json:"extraArgs,omitempty"`
}

// CacheBlendEngineStatus defines the observed state of CacheBlendEngine.
type CacheBlendEngineStatus struct {
	// phase is the overall phase of the CacheBlendEngine.
	// +optional
	Phase string `json:"phase,omitempty"`

	// observedGeneration is the most recent generation observed.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// desiredInstances is the number of desired instances.
	// +optional
	DesiredInstances int32 `json:"desiredInstances,omitempty"`

	// readyInstances is the number of ready instances.
	// +optional
	ReadyInstances int32 `json:"readyInstances,omitempty"`

	// endpoints lists per-node connection info.
	// +optional
	Endpoints []EndpointStatus `json:"endpoints,omitempty"`

	// conditions represent the current state of the CacheBlendEngine resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=cbe
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyInstances`
// +kubebuilder:printcolumn:name="Desired",type=integer,JSONPath=`.status.desiredInstances`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// CacheBlendEngine is the Schema for the cacheblendengines API.
type CacheBlendEngine struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of CacheBlendEngine.
	// +required
	Spec CacheBlendEngineSpec `json:"spec"`

	// status defines the observed state of CacheBlendEngine.
	// +optional
	Status CacheBlendEngineStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// CacheBlendEngineList contains a list of CacheBlendEngine.
type CacheBlendEngineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []CacheBlendEngine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CacheBlendEngine{}, &CacheBlendEngineList{})
}
