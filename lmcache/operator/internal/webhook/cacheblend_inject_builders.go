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

// Package webhook implements the CacheBlend mutating admission webhook that
// injects the lmcache-cacheblend vLLM plugin into opted-in vLLM pods (see
// design §7). This file holds the pure, side-effect-free mutation builders; the
// admission handler that orchestrates them lives in pod_injector.go.
package webhook

import (
	corev1 "k8s.io/api/core/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	// cbPluginVolumeName is the name of the shared emptyDir volume that the init
	// container stages the lmcache-cacheblend plugin tree into and the vLLM
	// container reads it back from (design §7 M1/M2/M3).
	cbPluginVolumeName = "cb-plugin"

	// cbPluginMountPath is the in-container path the cb-plugin volume mounts at,
	// in both the init container (read-write, the cp target) and the vLLM
	// container (read-only). It is also the vLLM container's prepended PYTHONPATH
	// entry, so the two stay in lockstep (design §9.5).
	cbPluginMountPath = "/cb-plugin"

	// cbSharedDirEnvName is the env var the payload init container reads to learn
	// where to copy the plugin tree (cacheblend-plugin docker/Dockerfile:22-29).
	cbSharedDirEnvName = "SHARED_DIR"

	// cbInitContainerName is the name of the injected payload init container.
	cbInitContainerName = "cb-plugin-stage"
)

// cbStaging is the CacheBlend injector's payload-staging parameters, consumed by
// the shared payloadStaging builders. The vLLM container's prepended PYTHONPATH
// equals cbPluginMountPath so the staged plugin is discoverable (design §7 M4).
var cbStaging = payloadStaging{
	volumeName:   cbPluginVolumeName,
	mountPath:    cbPluginMountPath,
	initName:     cbInitContainerName,
	sharedDirEnv: cbSharedDirEnvName,
}

// CacheBlend-required vLLM flag names and fixed values (design §7 M5). The
// CacheBlend matcher and connector hard-require these; several fail loudly,
// --no-async-scheduling fails silently (MoE garble).
const (
	cbFlagAttentionBackend = "--attention-backend"
	cbValAttentionBackend  = "CUSTOM"

	cbFlagKVTransferConfig = "--kv-transfer-config"

	cbFlagNoChunkedPrefill = "--no-enable-chunked-prefill"

	cbFlagBlockSize = "--block-size"
	cbValBlockSize  = "64"

	cbFlagPipelineParallelSize = "--pipeline-parallel-size"
	cbValPipelineParallelSize  = "1"

	cbFlagNoAsyncScheduling = "--no-async-scheduling"

	// cudagraph mode flags. Eager (default) forces --enforce-eager; full
	// decode-only enables decode graphs while never using full graphs in prefill
	// (design §4, §7 M5).
	cbFlagEnforceEager      = "--enforce-eager"
	cbFlagCompilationConfig = "--compilation-config"
)

// BuildCBPluginVolume returns the shared emptyDir volume the init container
// stages the plugin into and the vLLM container reads it back from (M1).
func BuildCBPluginVolume() corev1.Volume {
	return cbStaging.volume()
}

// BuildCBInitContainer returns the payload init container (M2). It mounts the
// cb-plugin volume read-write at /cb-plugin, sets SHARED_DIR=/cb-plugin, and runs
// the payload image's own ENTRYPOINT (busybox `cp -a`) with no command override.
//
// Parameters:
//   - payloadImage: the (possibly private) image that ships the unpacked
//     lmcache_cacheblend plugin tree under /payload.
//   - pullPolicy: the image pull policy for that image.
func BuildCBInitContainer(payloadImage string, pullPolicy corev1.PullPolicy) corev1.Container {
	return cbStaging.initContainer(payloadImage, pullPolicy)
}

// BuildCBVolumeMount returns the read-only mount of the cb-plugin volume added to
// the target vLLM container (M3).
func BuildCBVolumeMount() corev1.VolumeMount {
	return cbStaging.volumeMount()
}

// BuildCBPodEnv returns the env list for the target vLLM container with
// PYTHONPATH set to (or prepended with) /cb-plugin (M4). It is set on the
// container, never the pod, so every spawned worker inherits it; it never sets
// VLLM_PLUGINS (design §9.8). An existing PYTHONPATH is prepended, not replaced,
// so /cb-plugin:<existing> keeps the plugin discoverable without dropping the
// user's path entries.
//
// Parameters:
//   - existing: the target container's current env list (may be nil).
//
// Returns a new env list; the input is not mutated.
func BuildCBPodEnv(existing []corev1.EnvVar) []corev1.EnvVar {
	return cbStaging.prependPythonPath(existing)
}

// cudagraphArgs returns the cudagraph-mode flag set for the given mode. "eager"
// (and any unrecognized value) maps to --enforce-eager; "full_decode_only"
// enables decode-only CUDA graphs without ever using full graphs in prefill;
// "piecewise" enables piecewise capture (no --enforce-eager).
func cudagraphArgs(cudagraph string) []string {
	switch cudagraph {
	case lmcachev1alpha1.CudagraphPiecewise:
		// Piecewise graph capture: do not force eager. vLLM's default
		// compilation already does piecewise capture, so no extra flag is
		// emitted; we simply omit --enforce-eager.
		return nil
	case lmcachev1alpha1.CudagraphFullDecodeOnly:
		// Decode-only full graphs: enable cudagraph for decode but never full
		// graphs in prefill (CacheBlend re-RoPE happens in prefill).
		return []string{cbFlagCompilationConfig, `{"cudagraph_mode":"FULL_DECODE_ONLY"}`}
	default:
		// eager (default): force --enforce-eager.
		return []string{cbFlagEnforceEager}
	}
}

// BuildCBArgs returns the target vLLM container's args with the CacheBlend
// required flag set applied (M5). Each flag is appended-or-replaced (design §9.1):
// a user-supplied --flag value (in either "--flag v" or "--flag=v" form) is
// overwritten in place rather than duplicated, and a flag the user never set is
// appended.
//
// The --kv-transfer-config flag is included here only when the caller passes a
// non-empty kvTransferConfigJSON; the handler skips it (and stamps a reason)
// when the user already supplied their own --kv-transfer-config (design §9.2),
// in which case it passes "" so this builder leaves the user's value untouched.
//
// Parameters:
//   - existingArgs: the target container's current args (may be nil).
//   - kvTransferConfigJSON: the CBKVConnector JSON from the engine's connection
//     ConfigMap, or "" to skip injecting/replacing --kv-transfer-config.
//   - cudagraph: the cudagraph mode (eager|piecewise|full_decode_only).
//
// Returns a new args slice; the input is not mutated.
func BuildCBArgs(existingArgs []string, kvTransferConfigJSON, cudagraph string) []string {
	args := make([]string, len(existingArgs))
	copy(args, existingArgs)

	args = applyArg(args, cbFlagAttentionBackend, cbValAttentionBackend)
	if kvTransferConfigJSON != "" {
		args = applyArg(args, cbFlagKVTransferConfig, kvTransferConfigJSON)
	}
	args = applyBareFlag(args, cbFlagNoChunkedPrefill)
	args = applyArg(args, cbFlagBlockSize, cbValBlockSize)
	args = applyArg(args, cbFlagPipelineParallelSize, cbValPipelineParallelSize)
	args = applyBareFlag(args, cbFlagNoAsyncScheduling)

	// cudagraphArgs returns either a single bare flag (--enforce-eager), a
	// [flag, value] pair (full_decode_only), or nil (piecewise). Apply with
	// append-or-replace semantics so a user's pre-existing value is overwritten.
	switch cg := cudagraphArgs(cudagraph); len(cg) {
	case 1:
		args = applyBareFlag(args, cg[0])
	case 2:
		args = applyArg(args, cg[0], cg[1])
	}

	return args
}

// applyArg appends-or-replaces a "--flag value" pair in args (design §9.1). It
// recognizes both the two-token form ["--flag", "value"] and the single-token
// form ["--flag=value"]; a match is overwritten in place (preserving the form),
// otherwise the two-token pair is appended. Returns a new slice.
func applyArg(args []string, flag, value string) []string {
	eqPrefix := flag + "="
	for i := range len(args) {
		if args[i] == flag {
			// Two-token form: overwrite the following value token if present,
			// else append the value after the flag.
			if i+1 < len(args) {
				args[i+1] = value
				return args
			}
			return append(args, value)
		}
		if len(args[i]) >= len(eqPrefix) && args[i][:len(eqPrefix)] == eqPrefix {
			// Single-token --flag=value form: overwrite in place.
			args[i] = eqPrefix + value
			return args
		}
	}
	return append(args, flag, value)
}

// applyBareFlag appends a valueless flag (e.g. --no-enable-chunked-prefill) if it
// is not already present in either bare or --flag=... form. Returns a new slice.
func applyBareFlag(args []string, flag string) []string {
	eqPrefix := flag + "="
	for _, a := range args {
		if a == flag {
			return args
		}
		if len(a) >= len(eqPrefix) && a[:len(eqPrefix)] == eqPrefix {
			return args
		}
	}
	return append(args, flag)
}

// MergeImagePullSecrets returns existing with injected appended, deduplicated by
// secret name (M7). A secret already present in existing is not duplicated, and
// the order of existing is preserved with new secrets appended in injected order.
// Returns a new slice; the inputs are not mutated.
func MergeImagePullSecrets(
	existing, injected []corev1.LocalObjectReference,
) []corev1.LocalObjectReference {
	seen := make(map[string]struct{}, len(existing))
	out := make([]corev1.LocalObjectReference, 0, len(existing)+len(injected))
	for _, ref := range existing {
		out = append(out, ref)
		seen[ref.Name] = struct{}{}
	}
	for _, ref := range injected {
		if _, ok := seen[ref.Name]; ok {
			continue
		}
		seen[ref.Name] = struct{}{}
		out = append(out, ref)
	}
	return out
}
