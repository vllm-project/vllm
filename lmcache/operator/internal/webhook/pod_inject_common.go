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

// Package webhook implements the pod-mutating admission webhooks that wire
// opted-in vLLM pods to an engine (CacheBlend or LMCache). This file holds the
// machinery shared by both injectors: the annotation key set, the opt-in /
// idempotency gate, the connection-ConfigMap + target-container resolution, and
// the skip / stamp / patch responses. The two injectors differ only in which
// engine they read and which mutations they apply (cacheblend_pod_injector.go,
// lmcache_pod_injector.go).
package webhook

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/LMCache/LMCache/internal/resources"
)

// valueTrue is the boolean-true string stamped on the injected guard and used as
// the opt-in value of each injector's inject label.
const valueTrue = "true"

// kvTransferConfigDataKey is the key within the <engine>-connection ConfigMap's
// Data map that holds the kv-transfer-config JSON. It must match the key written
// by resources.buildConnectionConfigMapCore.
const kvTransferConfigDataKey = "kv-transfer-config.json"

// Skip-reason values shared by both injectors (stamped on the per-injector
// skip-reason annotation; design §8). CacheBlend adds SkipReasonPayloadImageUnset.
const (
	// SkipReasonEngineNotFound is stamped when the named engine's connection
	// ConfigMap (or, for CacheBlend, the engine CR) does not exist (fail-open).
	SkipReasonEngineNotFound = "engine-not-found"

	// SkipReasonCommandOverride is stamped when the target container overrides
	// command, so appended args may never reach `vllm serve`.
	SkipReasonCommandOverride = "command-override"

	// SkipReasonKVTransferConfigPresent is stamped when the user already supplies
	// --kv-transfer-config; the webhook does not clobber their structured JSON.
	SkipReasonKVTransferConfigPresent = "kv-transfer-config-present"

	// SkipReasonTargetContainerNotFound is stamped when the requested target
	// container names a container that does not exist on the pod.
	SkipReasonTargetContainerNotFound = "target-container-not-found"
)

// injectionKeys is one injector's annotation key set. It mirrors the same four
// concerns for each injector (CacheBlend / LMCache) so the shared gate, skip,
// and stamp helpers are key-set agnostic.
type injectionKeys struct {
	// engine binds a pod to an engine in the same namespace. Its presence is the
	// opt-in signal; its value is the engine name.
	engine string
	// container optionally names the target vLLM container (empty = first).
	container string
	// injected is the idempotency guard stamped after a successful injection.
	injected string
	// skipReason records why injection was skipped (fail-open).
	skipReason string
}

// gate decodes the pod and applies the shared idempotency + opt-in gates. When
// handled is true the caller must return resp unchanged: either a decode error,
// an unchanged Allow for an already-injected or non-opted-in pod. Otherwise it
// returns the decoded pod, the bound engine name, and the lookup namespace
// (req.Namespace, falling back to pod.Namespace which may be empty on CREATE).
func (k injectionKeys) gate(
	decoder admission.Decoder,
	req admission.Request,
) (pod *corev1.Pod, engineName, namespace string, resp admission.Response, handled bool) {
	pod = &corev1.Pod{}
	if err := decoder.Decode(req, pod); err != nil {
		return nil, "", "", admission.Errored(http.StatusBadRequest, err), true
	}
	if pod.Annotations[k.injected] == valueTrue {
		return pod, "", "", admission.Allowed("already injected"), true
	}
	engineName = strings.TrimSpace(pod.Annotations[k.engine])
	if engineName == "" {
		return pod, "", "", admission.Allowed("not opted in"), true
	}
	namespace = req.Namespace
	if namespace == "" {
		namespace = pod.Namespace
	}
	return pod, engineName, namespace, admission.Response{}, false
}

// prepareInjection performs the resolution shared by both injectors: read the
// engine's <engine>-connection ConfigMap (existence gate, fail-open), resolve
// the target container (annotation override > specDefault > first), and apply
// the command-override gate. When ok is false the caller must return resp (a
// skip + stamp, or an internal error). Otherwise it returns the kv-transfer
// JSON and the target container index.
//
// Parameters:
//   - specDefault: the engine's default target container name (nil = first); the
//     per-pod container annotation overrides it.
func prepareInjection(
	ctx context.Context,
	c client.Client,
	req admission.Request,
	pod *corev1.Pod,
	keys injectionKeys,
	engineName, namespace string,
	specDefault *string,
) (kvJSON string, targetIdx int, resp admission.Response, ok bool) {
	log := ctrl.LoggerFrom(ctx)

	connCM := &corev1.ConfigMap{}
	connName := resources.ConnectionConfigMapName(engineName)
	if err := c.Get(ctx, types.NamespacedName{Name: connName, Namespace: namespace}, connCM); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Skipped injection: connection ConfigMap not found",
				"engine", engineName, "configMap", connName, "namespace", namespace)
			return "", 0, keys.skip(req, pod, SkipReasonEngineNotFound), false
		}
		return "", 0, admission.Errored(http.StatusInternalServerError, err), false
	}
	kvJSON = connCM.Data[kvTransferConfigDataKey]

	idx, found := resolveTargetContainer(pod, specDefault, pod.Annotations[keys.container])
	if !found {
		log.Info("Skipped injection: target container not found",
			"engine", engineName, "annotationContainer", pod.Annotations[keys.container])
		return "", 0, keys.skip(req, pod, SkipReasonTargetContainerNotFound), false
	}

	if len(pod.Spec.Containers[idx].Command) > 0 {
		log.Info("Skipped injection: target container overrides command",
			"engine", engineName, "container", pod.Spec.Containers[idx].Name)
		return "", 0, keys.skip(req, pod, SkipReasonCommandOverride), false
	}
	return kvJSON, idx, admission.Response{}, true
}

// stampInjected stamps the idempotency guard (and the kv-transfer-config-present
// skip reason when the user supplied their own --kv-transfer-config), then
// returns the success patch response. Callers apply their mutations first.
func (k injectionKeys) stampInjected(
	req admission.Request,
	pod *corev1.Pod,
	userHasKVTransferConfig bool,
) admission.Response {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[k.injected] = valueTrue
	if userHasKVTransferConfig {
		pod.Annotations[k.skipReason] = SkipReasonKVTransferConfigPresent
	}
	return patchResponse(req, pod)
}

// skip stamps the given skip reason on the pod (without injecting) and returns
// an Allowed patch response. The pod is still admitted (fail-open).
func (k injectionKeys) skip(req admission.Request, pod *corev1.Pod, reason string) admission.Response {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[k.skipReason] = reason
	return patchResponse(req, pod)
}

// patchResponse marshals the mutated pod and returns a JSON patch response
// against the original raw object (req.Object.Raw).
func patchResponse(req admission.Request, pod *corev1.Pod) admission.Response {
	marshaled, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaled)
}

// resolveTargetContainer returns the index of the container to inject into and
// whether one was found. The per-pod annotation override (annotationName) takes
// precedence over the engine's default; an empty selection falls back to the
// first container. A non-empty name that matches no container yields ok=false.
//
// Parameters:
//   - pod: the decoded pod.
//   - specDefault: the engine's default target container (nil/"" = first).
//   - annotationName: the per-pod container-override annotation value.
func resolveTargetContainer(
	pod *corev1.Pod,
	specDefault *string,
	annotationName string,
) (int, bool) {
	if len(pod.Spec.Containers) == 0 {
		return 0, false
	}

	name := strings.TrimSpace(annotationName)
	if name == "" && specDefault != nil {
		name = strings.TrimSpace(*specDefault)
	}
	if name == "" {
		return 0, true
	}
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			return i, true
		}
	}
	return 0, false
}

// argsHasFlag reports whether args already carries the given flag in either the
// two-token "--flag value" form or the single-token "--flag=value" form.
func argsHasFlag(args []string, flag string) bool {
	eqPrefix := flag + "="
	for _, a := range args {
		if a == flag {
			return true
		}
		if strings.HasPrefix(a, eqPrefix) {
			return true
		}
	}
	return false
}
