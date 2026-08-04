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

package webhook

import (
	corev1 "k8s.io/api/core/v1"
)

// pythonPathEnvName is the standard Python module search-path env var prepended
// on the target vLLM container so it (and every spawned worker subprocess)
// discovers a staged payload tree.
const pythonPathEnvName = "PYTHONPATH"

// payloadStaging parameterizes the init-container code-staging mutation shared by
// the CacheBlend and LMCache injectors. The pattern is identical for both: a
// shared emptyDir volume, a payload init container that copies an unpacked
// package tree into it, a read-only mount of that volume on the target vLLM
// container, and a PYTHONPATH entry so vLLM imports the staged tree. The two
// injectors differ only in the names/paths they stage under, so they share this
// machinery and supply their own constants (cbStaging, lmcStaging).
type payloadStaging struct {
	// volumeName is the shared emptyDir volume the init container stages the
	// payload tree into and the vLLM container reads it back from.
	volumeName string
	// mountPath is the in-container path the volume mounts at — read-write in the
	// init container (the copy target) and read-only in the vLLM container. It is
	// also the PYTHONPATH entry, so the two must stay in lockstep.
	mountPath string
	// initName is the name of the injected payload init container.
	initName string
	// sharedDirEnv is the env var the payload image reads to learn where to copy
	// the tree (its busybox `cp -a /payload $SHARED_DIR` ENTRYPOINT).
	sharedDirEnv string
}

// volume returns the shared emptyDir volume the init container stages the
// payload into and the vLLM container reads it back from.
func (s payloadStaging) volume() corev1.Volume {
	return corev1.Volume{
		Name: s.volumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
}

// initContainer returns the payload init container. It mounts the shared volume
// read-write at mountPath, sets the sharedDirEnv to that path, and runs the
// payload image's own ENTRYPOINT (busybox `cp -a`) with no command override.
//
// Parameters:
//   - image: the (possibly private) image that ships the unpacked package tree
//     under /payload.
//   - pullPolicy: the image pull policy for that image.
func (s payloadStaging) initContainer(image string, pullPolicy corev1.PullPolicy) corev1.Container {
	return corev1.Container{
		Name:            s.initName,
		Image:           image,
		ImagePullPolicy: pullPolicy,
		Env: []corev1.EnvVar{
			{Name: s.sharedDirEnv, Value: s.mountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      s.volumeName,
				MountPath: s.mountPath,
				ReadOnly:  false,
			},
		},
	}
}

// volumeMount returns the read-only mount of the shared volume added to the
// target vLLM container.
func (s payloadStaging) volumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      s.volumeName,
		MountPath: s.mountPath,
		ReadOnly:  true,
	}
}

// prependPythonPath returns the target vLLM container's env with mountPath set as
// (or prepended to) PYTHONPATH. It is set on the container, never the pod, so
// every spawned worker inherits it. An existing PYTHONPATH is prepended, not
// replaced, so <mountPath>:<existing> keeps the staged tree discoverable without
// dropping the user's path entries.
//
// Parameters:
//   - existing: the target container's current env list (may be nil).
//
// Returns a new env list; the input is not mutated.
func (s payloadStaging) prependPythonPath(existing []corev1.EnvVar) []corev1.EnvVar {
	out := make([]corev1.EnvVar, 0, len(existing)+1)
	found := false
	for _, e := range existing {
		if e.Name == pythonPathEnvName {
			found = true
			prepended := e
			if prepended.ValueFrom != nil {
				// A valueFrom PYTHONPATH cannot be string-prepended safely; in
				// that rare case overwrite with the staged path so the tree is
				// at least discoverable (the alternative is no payload at all).
				prepended.ValueFrom = nil
				prepended.Value = s.mountPath
			} else if prepended.Value == "" {
				prepended.Value = s.mountPath
			} else {
				prepended.Value = s.mountPath + ":" + prepended.Value
			}
			out = append(out, prepended)
			continue
		}
		out = append(out, e)
	}
	if !found {
		out = append(out, corev1.EnvVar{Name: pythonPathEnvName, Value: s.mountPath})
	}
	return out
}
