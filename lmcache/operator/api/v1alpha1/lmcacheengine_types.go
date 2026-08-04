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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Phase constants for LMCacheEngine status.
const (
	PhasePending  = "Pending"
	PhaseRunning  = "Running"
	PhaseDegraded = "Degraded"
	PhaseFailed   = "Failed"
)

const (
	GPUVendorNvidia = "nvidia"
	GPUVendorAMD    = "amd"
)

// Condition type constants.
const (
	ConditionAvailable         = "Available"
	ConditionAllInstancesReady = "AllInstancesReady"
	ConditionConfigValid       = "ConfigValid"
)

// ImageSpec defines the container image to use.
type ImageSpec struct {
	// repository is the container image repository.
	// +optional
	// +kubebuilder:default="lmcache/vllm-openai"
	Repository *string `json:"repository,omitempty"`

	// tag is the container image tag.
	// +optional
	// +kubebuilder:default="latest"
	Tag *string `json:"tag,omitempty"`

	// pullPolicy is the image pull policy.
	// +optional
	// +kubebuilder:default="IfNotPresent"
	// +kubebuilder:validation:Enum=Always;Never;IfNotPresent
	PullPolicy *string `json:"pullPolicy,omitempty"`
}

// LMCacheInjectionSpec configures optional lmcache code-payload staging for vLLM
// pods bound to this engine. When payloadImage is unset the webhook wires only
// the connection (--kv-transfer-config); when set it also copies the payload
// tree into the vLLM container and prepends PYTHONPATH so vLLM imports that
// lmcache instead of the one baked into its image, keeping the vLLM client and
// the engine server on one build.
type LMCacheInjectionSpec struct {
	// payloadImage is a purpose-built init-container image (repository/tag/
	// pullPolicy, like spec.image) that ships an unpacked lmcache tree under
	// /payload and copies it to $SHARED_DIR on start (see
	// docker/Dockerfile.payload); the vLLM container imports it via PYTHONPATH.
	//
	// repository has no usable default — the ImageSpec default
	// (lmcache/vllm-openai) is NOT a payload — so it must be set explicitly. For
	// private registries, imagePullSecrets must reference Secret(s) in the vLLM
	// pod's namespace.
	// +optional
	PayloadImage *ImageSpec `json:"payloadImage,omitempty"`

	// imagePullSecrets are appended to the vLLM pod's spec.imagePullSecrets so a
	// PRIVATE payload init-container image can pull. The referenced Secret(s)
	// must already exist in the vLLM pod's namespace; the operator does not copy
	// them cross-namespace.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// targetContainer is the name of the vLLM container to inject into. Empty
	// (the default) selects the first container; a per-pod annotation may
	// override it.
	// +optional
	TargetContainer *string `json:"targetContainer,omitempty"`
}

// ServerSpec defines server configuration mapping to server.py argparse.
type ServerSpec struct {
	// port is the server listening port.
	// +optional
	// +kubebuilder:default=5555
	// +kubebuilder:validation:Minimum=1024
	// +kubebuilder:validation:Maximum=65535
	Port *int32 `json:"port,omitempty"`

	// chunkSize is the token chunk size.
	// +optional
	// +kubebuilder:default=256
	ChunkSize *int32 `json:"chunkSize,omitempty"`

	// maxWorkers is the number of worker threads.
	// +optional
	// +kubebuilder:default=1
	MaxWorkers *int32 `json:"maxWorkers,omitempty"`

	// hashAlgorithm is the hash algorithm used for token hashing.
	// +optional
	// +kubebuilder:default="blake3"
	// +kubebuilder:validation:Enum=builtin;sha256_cbor;blake3
	HashAlgorithm *string `json:"hashAlgorithm,omitempty"`

	// httpPort is the HTTP frontend port (health checks, cache admin).
	// +optional
	// +kubebuilder:default=8080
	// +kubebuilder:validation:Minimum=1024
	// +kubebuilder:validation:Maximum=65535
	HTTPPort *int32 `json:"httpPort,omitempty"`
}

// L1BackendSpec defines the L1 memory cache configuration.
type L1BackendSpec struct {
	// sizeGB is the L1 cache size in gigabytes. Required, must be > 0.
	// The CRD-level constraint (exclusiveMinimum=0) rejects invalid values
	// at admission time so the controller never sees them; the in-Go
	// ValidateSpec keeps the same rule for defense in depth.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:ExclusiveMinimum=true
	SizeGB float64 `json:"sizeGB"`
}

// EvictionSpec defines the cache eviction configuration.
type EvictionSpec struct {
	// policy is the eviction policy. LRU or noop.
	// +optional
	// +kubebuilder:default="LRU"
	// +kubebuilder:validation:Enum=LRU;noop
	Policy *string `json:"policy,omitempty"`

	// triggerWatermark is the cache usage ratio that triggers eviction.
	// +optional
	// +kubebuilder:default=0.8
	TriggerWatermark *float64 `json:"triggerWatermark,omitempty"`

	// evictionRatio is the fraction of cache to evict when triggered.
	// +optional
	// +kubebuilder:default=0.2
	EvictionRatio *float64 `json:"evictionRatio,omitempty"`
}

// ServiceMonitorSpec defines Prometheus ServiceMonitor configuration.
type ServiceMonitorSpec struct {
	// enabled controls whether a ServiceMonitor CR is created.
	// +optional
	// +kubebuilder:default=false
	Enabled *bool `json:"enabled,omitempty"`

	// interval is the Prometheus scrape interval.
	// +optional
	// +kubebuilder:default="30s"
	Interval *string `json:"interval,omitempty"`

	// labels are additional labels added to the ServiceMonitor.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
}

// PrometheusSpec defines Prometheus monitoring configuration.
type PrometheusSpec struct {
	// enabled controls whether Prometheus metrics are exposed.
	// +optional
	// +kubebuilder:default=true
	Enabled *bool `json:"enabled,omitempty"`

	// port is the Prometheus metrics port.
	// +optional
	// +kubebuilder:default=9090
	Port *int32 `json:"port,omitempty"`

	// serviceMonitor configures the Prometheus ServiceMonitor.
	// +optional
	ServiceMonitor *ServiceMonitorSpec `json:"serviceMonitor,omitempty"`
}

// L2BackendSpec defines the L2 storage backend.
// Exactly one of RESP or Raw must be set.
type L2BackendSpec struct {
	// resp configures a Redis/Valkey RESP L2 adapter backed by the
	// native C++ connector.
	// +optional
	RESP *RESPL2AdapterSpec `json:"resp,omitempty"`

	// raw is an escape hatch for adapter types not yet natively
	// supported by the operator (e.g. nixl_store, fs, mock).
	// The JSON is passed through to --l2-adapter as-is.
	// +optional
	Raw *RawL2AdapterSpec `json:"raw,omitempty"`

	// storePolicy controls how keys flow from L1 to L2.
	// "default" stores all keys to the adapter and keeps L1.
	// "skip_l1" stores all keys to the adapter and deletes them from L1
	// (buffer-only mode — pair with eviction.policy=noop).
	// +optional
	// +kubebuilder:default="default"
	// +kubebuilder:validation:Enum=default;skip_l1
	StorePolicy *string `json:"storePolicy,omitempty"`

	// prefetchPolicy controls how keys flow from L2 back to L1 on
	// cache misses. "default" picks the first adapter that has the key.
	// +optional
	// +kubebuilder:default="default"
	// +kubebuilder:validation:Enum=default;retain
	PrefetchPolicy *string `json:"prefetchPolicy,omitempty"`

	// prefetchMaxInFlight limits the number of concurrent prefetch
	// (L2→L1 load) requests, preventing excessive L1 memory pressure.
	// +optional
	// +kubebuilder:default=8
	// +kubebuilder:validation:Minimum=1
	PrefetchMaxInFlight *int32 `json:"prefetchMaxInFlight,omitempty"`

	// serde configures a transform applied to KV bytes on their way to
	// and from the L2 adapter (whichever of resp or raw is configured).
	// Exactly one serde type must be set.
	// +optional
	Serde *L2SerdeSpec `json:"serde,omitempty"`
}

// L2SerdeSpec selects and configures the serde for the L2 backend. It
// renders to the "serde" sub-dict of the --l2-adapter JSON. Exactly one
// serde type must be set.
type L2SerdeSpec struct {
	// aesgcm enables at-rest encryption of KV bytes in the L2 tier,
	// keyed per cache_salt. L1 (host RAM) and L0 (GPU) remain plaintext.
	// +optional
	AESGCM *AESGCMSerdeSpec `json:"aesgcm,omitempty"`
}

// AESGCMSerdeSpec configures the aesgcm encryption serde. The operator
// mounts the referenced master-key Secret into the engine pods.
type AESGCMSerdeSpec struct {
	// masterKeySecretRef names a user-created Secret in the engine's
	// namespace holding the master key under the "master" data key. The
	// Secret is mounted directly into the engine pods; the reference is
	// same-namespace only, so creating an engine CR cannot be used to
	// read Secrets from other namespaces. The operator never generates
	// key material.
	MasterKeySecretRef corev1.LocalObjectReference `json:"masterKeySecretRef"`

	// keyProvider selects how per-cache_salt keys are obtained from the
	// master key. Only "hkdf" (HKDF-SHA256 derivation) is implemented.
	// +optional
	// +kubebuilder:default="hkdf"
	// +kubebuilder:validation:Enum=hkdf
	KeyProvider *string `json:"keyProvider,omitempty"`

	// aesBits selects the AES key size (128 or 256).
	// +optional
	// +kubebuilder:default=128
	// +kubebuilder:validation:Enum=128;256
	AESBits *int32 `json:"aesBits,omitempty"`
}

// RESPL2AdapterSpec configures a RESP (Redis/Valkey) L2 adapter.
type RESPL2AdapterSpec struct {
	// host is the Redis/Valkey server hostname or IP.
	Host string `json:"host"`

	// port is the Redis/Valkey server port.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port int32 `json:"port"`

	// numWorkers is the number of C++ worker threads for I/O.
	// +optional
	// +kubebuilder:default=8
	// +kubebuilder:validation:Minimum=1
	NumWorkers *int32 `json:"numWorkers,omitempty"`

	// maxCapacityGB is the max L2 capacity in GB for usage tracking
	// and eviction. 0 means disabled.
	// +optional
	// +kubebuilder:default=0
	MaxCapacityGB *float64 `json:"maxCapacityGB,omitempty"`

	// authSecretRef is a reference to a Secret containing "username"
	// and "password" keys for Redis authentication.
	// The Secret may live in a different namespace; the operator will
	// create a managed copy in the LMCacheEngine's namespace.
	// The credentials are injected via environment variables so they
	// do not appear in container args or kubectl describe output.
	// +optional
	AuthSecretRef *SecretReference `json:"authSecretRef,omitempty"`
}

// SecretReference is a reference to a Secret that supports cross-namespace access.
type SecretReference struct {
	// name is the name of the Secret.
	Name string `json:"name"`

	// namespace is the namespace of the Secret.
	// If empty, defaults to the namespace of the LMCacheEngine resource.
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// RawL2AdapterSpec is a pass-through escape hatch for adapter types
// not natively supported by the operator. The type and config fields
// are merged into a flat JSON object and passed to --l2-adapter.
type RawL2AdapterSpec struct {
	// type is the adapter type name (e.g. "nixl_store", "fs", "mock").
	Type string `json:"type"`

	// config is type-specific configuration as a free-form map.
	// +optional
	Config map[string]apiextensionsv1.JSON `json:"config,omitempty"`
}

// CoordinatorConnectionSpec configures how an engine server registers with an
// MP coordinator. It maps to the server's coordinator-client flags
// (lmcache/v1/multiprocess/config.py: add_coordinator_args). Exactly one of
// ref or url must be set.
type CoordinatorConnectionSpec struct {
	// ref names an LMCacheCoordinator in the same namespace. The operator
	// resolves it to the coordinator's in-cluster Service URL.
	// +optional
	Ref *corev1.LocalObjectReference `json:"ref,omitempty"`

	// url is an explicit coordinator base URL (e.g. http://coordinator:9300),
	// used to target a coordinator the operator does not manage.
	// +optional
	URL *string `json:"url,omitempty"`

	// advertiseIP is the IP the coordinator should reach this server at.
	//
	// DO NOT SET THIS IN ALMOST EVERY CASE. When unset, the server's pod IP is
	// injected automatically via the downward API, which is the correct value
	// for normal in-cluster deployments. Only set this if you know exactly what
	// you are doing -- e.g. the coordinator runs outside the cluster and must
	// reach the server through a specific externally-routable address. An
	// incorrect value silently breaks coordinator-to-server connectivity.
	// +optional
	AdvertiseIP *string `json:"advertiseIP,omitempty"`

	// heartbeatInterval is the seconds between heartbeats; must be > 0.
	// +optional
	// +kubebuilder:default=5
	HeartbeatInterval *float64 `json:"heartbeatInterval,omitempty"`

	// l2EventReporting enables reporting L2 store/lookup events to the
	// coordinator for fleet-wide usage tracking and eviction.
	// +optional
	// +kubebuilder:default=false
	L2EventReporting *bool `json:"l2EventReporting,omitempty"`

	// l2EventFlushInterval is the seconds between L2 event flush attempts; must
	// be > 0.
	// +optional
	// +kubebuilder:default=1
	L2EventFlushInterval *float64 `json:"l2EventFlushInterval,omitempty"`
}

// LMCacheEngineSpec defines the desired state of LMCacheEngine.
type LMCacheEngineSpec struct {
	// gpuVendor selects the GPU vendor. "nvidia" (default) requires the NVIDIA
	// GPU Operator's "nvidia" RuntimeClass; "amd" runs on the default container
	// runtime.
	// +optional
	// +kubebuilder:default="nvidia"
	// +kubebuilder:validation:Enum=nvidia;amd
	GPUVendor *string `json:"gpuVendor,omitempty"`

	// image defines the container image to use.
	// +optional
	Image *ImageSpec `json:"image,omitempty"`

	// imagePullSecrets is a list of references to secrets for pulling the image.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// server defines server configuration.
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

	// injection defines the defaults the LMCache mutating webhook reads for pods
	// bound to this engine. When unset, the webhook wires only the connection;
	// set injection.payloadImage to also stage an lmcache code payload.
	// +optional
	Injection *LMCacheInjectionSpec `json:"injection,omitempty"`

	// resourceOverrides allows overriding auto-computed resource requirements.
	// +optional
	ResourceOverrides *corev1.ResourceRequirements `json:"resourceOverrides,omitempty"`

	// logLevel is the log level for the LMCache server.
	// +optional
	// +kubebuilder:default="INFO"
	// +kubebuilder:validation:Enum=DEBUG;INFO;WARNING;ERROR
	LogLevel *string `json:"logLevel,omitempty"`

	// nodeSelector determines which nodes get an LMCache instance.
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

	// hostNetwork runs the pod in the host's network namespace. When true the
	// operator also sets dnsPolicy to ClusterFirstWithHostNet so cluster DNS
	// still works. Default: false.
	// +optional
	// +kubebuilder:default=false
	HostNetwork *bool `json:"hostNetwork,omitempty"`

	// privileged runs the engine container in privileged mode. On some clusters
	// this is required for the engine to see all node GPUs (for CUDA IPC) without
	// claiming any via the nvidia.com/gpu device plugin; on many clusters
	// NVIDIA_VISIBLE_DEVICES=all already grants that visibility without it, so it
	// defaults to false.
	// +optional
	// +kubebuilder:default=false
	Privileged *bool `json:"privileged,omitempty"`

	// extraArgs are additional CLI flags appended to the server command.
	// They are appended last and can override any auto-generated flag.
	// +optional
	ExtraArgs []string `json:"extraArgs,omitempty"`
}

// EndpointStatus represents a single LMCache instance endpoint.
type EndpointStatus struct {
	// nodeName is the name of the node running this instance.
	NodeName string `json:"nodeName"`

	// hostIP is the IP address of the host.
	HostIP string `json:"hostIP"`

	// podName is the name of the pod.
	PodName string `json:"podName"`

	// port is the server port.
	Port int32 `json:"port"`

	// metricsPort is the Prometheus metrics port.
	MetricsPort int32 `json:"metricsPort"`

	// ready indicates whether this instance is ready.
	Ready bool `json:"ready"`
}

// LMCacheEngineStatus defines the observed state of LMCacheEngine.
type LMCacheEngineStatus struct {
	// phase is the overall phase of the LMCacheEngine.
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

	// conditions represent the current state of the LMCacheEngine resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=lmc
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyInstances`
// +kubebuilder:printcolumn:name="Desired",type=integer,JSONPath=`.status.desiredInstances`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// LMCacheEngine is the Schema for the lmcacheengines API.
type LMCacheEngine struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of LMCacheEngine.
	// +required
	Spec LMCacheEngineSpec `json:"spec"`

	// status defines the observed state of LMCacheEngine.
	// +optional
	Status LMCacheEngineStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// LMCacheEngineList contains a list of LMCacheEngine.
type LMCacheEngineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []LMCacheEngine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LMCacheEngine{}, &LMCacheEngineList{})
}
