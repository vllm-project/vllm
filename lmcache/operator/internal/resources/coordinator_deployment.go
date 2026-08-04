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
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	// coordinatorSubcommand is the `lmcache coordinator` subcommand that starts
	// the fleet coordinator HTTP server.
	coordinatorSubcommand = "coordinator"

	// coordinatorDefaultImageRepo is the default coordinator image repository.
	// The coordinator runs the same lmcache binary as the engines.
	coordinatorDefaultImageRepo = "lmcache/vllm-openai"

	// coordinatorPortName is the name of the coordinator's HTTP port on the
	// container and Service.
	coordinatorPortName = "http"

	// coordinatorDefaultPort is the default HTTP port (mirrors
	// MPCoordinatorConfig.port).
	coordinatorDefaultPort int32 = 9300

	// coordinatorHealthPath is the liveness/readiness probe path served by the
	// coordinator (lmcache/v1/mp_coordinator/http_apis/health_api.py).
	coordinatorHealthPath = "/healthz"

	// coordinatorComponentLabel is the app.kubernetes.io/component value for
	// coordinator-owned resources, distinguishing them from cache-engine pods.
	coordinatorComponentLabel = "coordinator"
)

// CoordinatorSelectorLabels returns the immutable label subset used for the
// coordinator's pod/Service selectors. It carries a distinct
// app.kubernetes.io/name so a coordinator never cross-selects engine pods that
// happen to share its name.
func CoordinatorSelectorLabels(name string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "lmcache-coordinator",
		"app.kubernetes.io/instance":   name,
		"app.kubernetes.io/managed-by": "lmcache-operator",
	}
}

// CoordinatorStandardLabels returns the full label set for coordinator-owned
// resources.
func CoordinatorStandardLabels(name string) map[string]string {
	labels := CoordinatorSelectorLabels(name)
	labels["app.kubernetes.io/component"] = coordinatorComponentLabel
	return labels
}

// CoordinatorPort returns the coordinator's HTTP port, applying the default.
func CoordinatorPort(spec *lmcachev1alpha1.LMCacheCoordinatorSpec) int32 {
	return derefInt32(spec.Port, coordinatorDefaultPort)
}

// BuildCoordinatorArgs maps the LMCacheCoordinatorSpec config fields to the
// `lmcache coordinator` CLI flags. User-supplied extraArgs are appended last so
// they retain override precedence.
//
// The global-CacheBlend knobs (blend_chunk_size / blend_probe_stride) are
// rendered via their `--blend-chunk-size` / `--blend-probe-stride` flags only
// when the spec sets them explicitly. When unset, the operator omits the flags
// and the coordinator image falls back to its own MPCoordinatorConfig defaults
// (256 / 1) -- this keeps the operator compatible with images whose CLI
// predates these flags. blendChunkSize MUST equal the chunk size the blend
// servers use.
func BuildCoordinatorArgs(spec *lmcachev1alpha1.LMCacheCoordinatorSpec) []string {
	// 7 always-on two-token flags, up to 2 optional blend flags, plus extras.
	args := make([]string, 0, 18+len(spec.ExtraArgs))
	args = append(args,
		"--host", derefString(spec.Host, "0.0.0.0"),
		"--port", fmt.Sprintf("%d", CoordinatorPort(spec)),
		"--instance-timeout", formatFloat(derefFloat64(spec.InstanceTimeout, 30.0)),
		"--health-check-interval", formatFloat(derefFloat64(spec.HealthCheckInterval, 10.0)),
		"--eviction-check-interval", formatFloat(derefFloat64(spec.EvictionCheckInterval, 5.0)),
		"--eviction-ratio", formatFloat(derefFloat64(spec.EvictionRatio, 0.2)),
		"--trigger-watermark", formatFloat(derefFloat64(spec.TriggerWatermark, 1.0)),
	)
	if spec.BlendChunkSize != nil {
		args = append(args, "--blend-chunk-size", fmt.Sprintf("%d", *spec.BlendChunkSize))
	}
	if spec.BlendProbeStride != nil {
		args = append(args, "--blend-probe-stride", fmt.Sprintf("%d", *spec.BlendProbeStride))
	}
	args = append(args, spec.ExtraArgs...)
	return args
}

// formatFloat renders a float flag value with the minimal decimal
// representation (e.g. 30.0 -> "30", 0.2 -> "0.2").
func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// BuildCoordinatorDeployment constructs the Deployment for the given
// LMCacheCoordinator. Unlike the engine DaemonSets, the coordinator is a plain
// fleet-wide service: no GPU runtime class, hostIPC, or privileged container.
func BuildCoordinatorDeployment(coordinator *lmcachev1alpha1.LMCacheCoordinator) *appsv1.Deployment {
	spec := &coordinator.Spec
	name := coordinator.Name
	namespace := coordinator.Namespace

	selectorLabels := CoordinatorSelectorLabels(name)
	podLabels := MergeLabels(CoordinatorStandardLabels(name), spec.PodLabels)

	port := CoordinatorPort(spec)

	imgRepo := coordinatorDefaultImageRepo
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

	envVars := make([]corev1.EnvVar, 0, 1+len(spec.Env))
	envVars = append(envVars, corev1.EnvVar{
		Name:  "LMCACHE_LOG_LEVEL",
		Value: derefString(spec.LogLevel, "INFO"),
	})
	envVars = append(envVars, spec.Env...)

	containerPorts := []corev1.ContainerPort{
		{
			Name:          coordinatorPortName,
			ContainerPort: port,
			Protocol:      corev1.ProtocolTCP,
		},
	}
	if coordinatorPrometheusEnabled(spec) {
		containerPorts = append(containerPorts, corev1.ContainerPort{
			Name:          "metrics",
			ContainerPort: coordinatorMetricsPort(spec),
			Protocol:      corev1.ProtocolTCP,
		})
	}

	httpProbe := func(period, failure int32) *corev1.Probe {
		return &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: coordinatorHealthPath,
					Port: intstr.FromInt32(port),
				},
			},
			PeriodSeconds:    period,
			FailureThreshold: failure,
		}
	}
	startupProbe := httpProbe(5, 30)
	startupProbe.InitialDelaySeconds = 5
	livenessProbe := httpProbe(10, 3)
	readinessProbe := httpProbe(5, 3)

	var resourceReqs corev1.ResourceRequirements
	if spec.ResourceOverrides != nil {
		resourceReqs = *spec.ResourceOverrides
	}

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    CoordinatorStandardLabels(name),
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: spec.Replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      podLabels,
					Annotations: spec.PodAnnotations,
				},
				Spec: corev1.PodSpec{
					ServiceAccountName: spec.ServiceAccountName,
					PriorityClassName:  spec.PriorityClassName,
					NodeSelector:       spec.NodeSelector,
					Affinity:           spec.Affinity,
					Tolerations:        spec.Tolerations,
					ImagePullSecrets:   spec.ImagePullSecrets,
					Containers: []corev1.Container{
						{
							Name:            coordinatorSubcommand,
							Image:           fmt.Sprintf("%s:%s", imgRepo, imgTag),
							ImagePullPolicy: imgPullPolicy,
							Command:         []string{lmcacheServerBinary, coordinatorSubcommand},
							Args:            BuildCoordinatorArgs(spec),
							Ports:           containerPorts,
							Env:             envVars,
							Resources:       resourceReqs,
							StartupProbe:    startupProbe,
							LivenessProbe:   livenessProbe,
							ReadinessProbe:  readinessProbe,
							VolumeMounts:    spec.VolumeMounts,
						},
					},
					Volumes: spec.Volumes,
				},
			},
		},
	}
}
