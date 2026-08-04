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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	// nvidiaRuntimeClass is the RuntimeClass name registered by the NVIDIA GPU
	// Operator; engine pods request it when gpuVendor is nvidia.
	nvidiaRuntimeClass = "nvidia"

	// lmcacheServerBinary is the entrypoint binary for the LMCache server inside
	// the engine image.
	lmcacheServerBinary = "/opt/venv/bin/lmcache"

	// serverSubcommand is the `lmcache server` subcommand that starts the engine.
	serverSubcommand = "server"

	// serverPortName is the name of the engine's serving port on the container
	// and the node-local Service.
	serverPortName = "server"

	// l2EncryptionKeyMountDir is where the L2 encryption master-key Secret is
	// mounted inside the engine container.
	l2EncryptionKeyMountDir = "/etc/lmcache/keys"

	// l2EncryptionKeyFileName is the required data key in the user-provided
	// master-key Secret and thus the file name under the mount dir.
	l2EncryptionKeyFileName = "master"

	// l2EncryptionKeyVolumeName is the pod volume name for the master-key mount.
	l2EncryptionKeyVolumeName = "l2-master-key"
)

// l2EncryptionKeyPath is the in-container path of the mounted L2 encryption
// master key, referenced as master_key_path in the serde config.
const l2EncryptionKeyPath = l2EncryptionKeyMountDir + "/" + l2EncryptionKeyFileName

// BuildDaemonSet constructs a DaemonSet for the given LMCacheEngine.
func BuildDaemonSet(engine *lmcachev1alpha1.LMCacheEngine) *appsv1.DaemonSet {
	return buildDaemonSetCore(engine.Name, engine.Namespace, &engine.Spec, BuildContainerArgs(&engine.Spec), "lmcache/vllm-openai")
}

// buildDaemonSetCore constructs the DaemonSet shared by the LMCacheEngine and
// CacheBlendEngine controllers. It is the single source of truth for the
// GPU/security pod-template scaffolding (hostIPC, runtimeClassName, optional
// privileged (default false, via spec.Privileged), NVIDIA_VISIBLE_DEVICES,
// resources without a device-plugin GPU claim) so those settings cannot drift
// between the two engines.
//
// Parameters:
//   - name, namespace: the owning object's identity, used for labels and metadata.
//   - spec: the engine spec (LMCacheEngine and CacheBlendEngine reuse the same
//     shared sub-structs, so callers project the CacheBlendEngine spec into an
//     *LMCacheEngineSpec before calling).
//   - containerArgs: the fully serialized server CLI args (callers append any
//     engine-specific flags such as --engine-type before passing them in).
//   - defaultImageRepo: the container image repository to use when spec.Image
//     does not set one.
func buildDaemonSetCore(
	name, namespace string,
	spec *lmcachev1alpha1.LMCacheEngineSpec,
	containerArgs []string,
	defaultImageRepo string,
) *appsv1.DaemonSet {
	selectorLabels := SelectorLabels(name)
	podLabels := MergeLabels(StandardLabels(name), spec.PodLabels)
	podAnnotations := spec.PodAnnotations

	gpuVendor := derefString(spec.GPUVendor, lmcachev1alpha1.GPUVendorNvidia)
	var runtimeClassName *string
	if gpuVendor == lmcachev1alpha1.GPUVendorNvidia {
		rc := nvidiaRuntimeClass
		runtimeClassName = &rc
	}
	privileged := derefBool(spec.Privileged, false)

	serverPort := derefInt32(getServerPort(spec), 5555)
	imgRepo := defaultImageRepo
	imgTag := "latest"
	imgPullPolicy := corev1.PullIfNotPresent
	if spec.Image != nil {
		imgRepo = derefString(spec.Image.Repository, imgRepo)
		imgTag = derefString(spec.Image.Tag, imgTag)
		switch derefString(spec.Image.PullPolicy, "IfNotPresent") {
		case "Always":
			imgPullPolicy = corev1.PullAlways
		case "Never":
			imgPullPolicy = corev1.PullNever
		default:
			imgPullPolicy = corev1.PullIfNotPresent
		}
	}

	// Build env vars
	envVars := make([]corev1.EnvVar, 0, 5+len(spec.Env))
	envVars = append(envVars,
		corev1.EnvVar{
			Name:  "LMCACHE_LOG_LEVEL",
			Value: derefString(spec.LogLevel, "INFO"),
		},
	)
	if gpuVendor == lmcachev1alpha1.GPUVendorNvidia {
		// Expose all GPUs without consuming device plugin resources.
		// LMCache needs GPU visibility for CUDA IPC, not compute ownership.
		envVars = append(envVars,
			corev1.EnvVar{
				Name:  "NVIDIA_VISIBLE_DEVICES",
				Value: "all",
			},
			corev1.EnvVar{
				Name:  "NVIDIA_DRIVER_CAPABILITIES",
				Value: "all",
			},
		)
	}
	// Inject RESP auth credentials from Secret as env vars so they
	// don't appear in container args or kubectl describe output.
	// The DaemonSet references the local (same-namespace) managed copy
	// created by the controller via reconcileRESPAuthSecret.
	if spec.L2Backend != nil && spec.L2Backend.RESP != nil && spec.L2Backend.RESP.AuthSecretRef != nil {
		secretName := RESPAuthSecretName(name)
		optional := true
		envVars = append(envVars,
			corev1.EnvVar{
				Name: "LMCACHE_RESP_PASSWORD",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: secretName},
						Key:                  "password",
					},
				},
			},
			corev1.EnvVar{
				// Optional: if the managed secret has no "username" key
				// (password-only auth), this env var is simply not set.
				Name: "LMCACHE_RESP_USERNAME",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: secretName},
						Key:                  "username",
						Optional:             &optional,
					},
				},
			},
		)
	}
	// Inject the pod IP as the coordinator advertise address (downward API)
	// when registration is enabled and no explicit advertiseIP is set, so the
	// coordinator can reach this server. The server's --coordinator-advertise-ip
	// flag falls back to this env var.
	if spec.Coordinator != nil && derefString(spec.Coordinator.URL, "") != "" &&
		(spec.Coordinator.AdvertiseIP == nil || *spec.Coordinator.AdvertiseIP == "") {
		envVars = append(envVars, corev1.EnvVar{
			Name: "LMCACHE_COORDINATOR_ADVERTISE_IP",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		})
	}
	envVars = append(envVars, spec.Env...)

	// No emptyDir /dev/shm mount — hostIPC: true exposes the host's /dev/shm
	// directly. An emptyDir mount would shadow it and break CUDA IPC between
	// LMCache and vLLM pods (cudaIpcOpenMemHandle requires shared /dev/shm).
	volumes := append([]corev1.Volume{}, spec.Volumes...)
	volumeMounts := append([]corev1.VolumeMount{}, spec.VolumeMounts...)

	// Mount the L2 encryption master key (user-created Secret in the engine's
	// namespace) read-only at the path the serde config references. Only the
	// "master" data key is projected; a missing Secret or missing key fails
	// the mount with a pod event and self-heals once the Secret is fixed.
	if spec.L2Backend != nil && spec.L2Backend.Serde != nil && spec.L2Backend.Serde.AESGCM != nil {
		keyMode := int32(0o400)
		volumes = append(volumes, corev1.Volume{
			Name: l2EncryptionKeyVolumeName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName: spec.L2Backend.Serde.AESGCM.MasterKeySecretRef.Name,
					Items: []corev1.KeyToPath{
						{Key: l2EncryptionKeyFileName, Path: l2EncryptionKeyFileName},
					},
					DefaultMode: &keyMode,
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      l2EncryptionKeyVolumeName,
			MountPath: l2EncryptionKeyMountDir,
			ReadOnly:  true,
		})
	}

	// Build container args. Auth credentials are handled via env vars
	// (LMCACHE_RESP_USERNAME / LMCACHE_RESP_PASSWORD) injected above,
	// so no shell wrapper is needed.
	containerCommand := []string{
		lmcacheServerBinary,
		serverSubcommand,
	}

	// Probes
	tcpProbe := &corev1.TCPSocketAction{
		Port: intstr.FromInt32(serverPort),
	}

	startupProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       5,
		FailureThreshold:    30,
	}

	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		PeriodSeconds: 10,
	}

	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		PeriodSeconds: 5,
	}

	// Container ports
	httpPort := getHTTPPort(spec)
	containerPorts := []corev1.ContainerPort{
		{
			Name:          serverPortName,
			ContainerPort: serverPort,
			Protocol:      corev1.ProtocolTCP,
		},
		{
			Name:          "http",
			ContainerPort: httpPort,
			Protocol:      corev1.ProtocolTCP,
		},
	}

	// Add metrics port if prometheus is enabled
	promEnabled := true
	promPort := int32(9090)
	if spec.Prometheus != nil {
		promEnabled = derefBool(spec.Prometheus.Enabled, true)
		promPort = derefInt32(spec.Prometheus.Port, 9090)
	}
	if promEnabled {
		containerPorts = append(containerPorts, corev1.ContainerPort{
			Name:          "metrics",
			ContainerPort: promPort,
			Protocol:      corev1.ProtocolTCP,
		})
	}

	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    StandardLabels(name),
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      podLabels,
					Annotations: podAnnotations,
				},
				Spec: corev1.PodSpec{
					HostIPC:            true,
					HostNetwork:        derefBool(spec.HostNetwork, false),
					RuntimeClassName:   runtimeClassName,
					ServiceAccountName: spec.ServiceAccountName,
					PriorityClassName:  spec.PriorityClassName,
					NodeSelector:       spec.NodeSelector,
					Affinity:           spec.Affinity,
					Tolerations:        spec.Tolerations,
					ImagePullSecrets:   spec.ImagePullSecrets,
					Containers: []corev1.Container{
						{
							Name:            "lmcache",
							Image:           fmt.Sprintf("%s:%s", imgRepo, imgTag),
							ImagePullPolicy: imgPullPolicy,
							Command:         containerCommand,
							Args:            containerArgs,
							Ports:           containerPorts,
							Env:             envVars,
							Resources:       ComputeResources(spec),
							SecurityContext: &corev1.SecurityContext{
								Privileged: &privileged,
							},
							VolumeMounts:   volumeMounts,
							StartupProbe:   startupProbe,
							LivenessProbe:  livenessProbe,
							ReadinessProbe: readinessProbe,
						},
					},
					Volumes: volumes,
				},
			},
		},
	}

	if derefBool(spec.HostNetwork, false) {
		ds.Spec.Template.Spec.DNSPolicy = corev1.DNSClusterFirstWithHostNet
	}

	return ds
}
