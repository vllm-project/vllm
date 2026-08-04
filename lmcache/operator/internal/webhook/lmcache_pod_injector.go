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
	"context"
	"net/http"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// LMCache annotation keys the webhook reads and stamps. They mirror the
// CacheBlend keys (cacheblend_pod_injector.go) with an lmcache- discriminator so
// the two injectors never cross-fire on the same pod.
const (
	// LMCacheAnnotationEngine binds a pod to an LMCacheEngine in the same
	// namespace. Its presence is the opt-in signal; its value is the engine name.
	LMCacheAnnotationEngine = "lmcache.ai/lmcache-engine"

	// LMCacheAnnotationContainer optionally names the target vLLM container;
	// empty or absent selects the first container.
	LMCacheAnnotationContainer = "lmcache.ai/lmcache-container"

	// LMCacheAnnotationImagePullSecrets optionally overrides the engine's
	// injection.imagePullSecrets with a comma-separated list of Secret names
	// appended to the pod's spec.imagePullSecrets for the private payload image.
	LMCacheAnnotationImagePullSecrets = "lmcache.ai/lmcache-image-pull-secrets"

	// LMCacheAnnotationInjected is the idempotency guard stamped after a
	// successful injection; a re-admitted pod carrying it is allowed unchanged.
	LMCacheAnnotationInjected = "lmcache.ai/lmcache-injected"

	// LMCacheAnnotationSkipReason records why injection was skipped (fail-open).
	LMCacheAnnotationSkipReason = "lmcache.ai/lmcache-skip-reason"
)

// LMCacheLabelInject is the opt-in label the MutatingWebhookConfiguration's
// objectSelector matches (mutating_webhook_selectors_patch.yaml). It gates which
// pods reach the webhook; the handler itself gates on LMCacheAnnotationEngine.
const LMCacheLabelInject = "lmcache.ai/lmcache-inject"

// lmCacheKeys is the LMCache injector's annotation key set, consumed by the
// shared gate / skip / stamp helpers.
var lmCacheKeys = injectionKeys{
	engine:     LMCacheAnnotationEngine,
	container:  LMCacheAnnotationContainer,
	injected:   LMCacheAnnotationInjected,
	skipReason: LMCacheAnnotationSkipReason,
}

// +kubebuilder:webhook:path=/mutate-lmcache--v1-pod,mutating=true,failurePolicy=ignore,sideEffects=None,groups="",resources=pods,verbs=create,versions=v1,name=mlmcachepod.lmcache.ai,admissionReviewVersions=v1,reinvocationPolicy=Never

// LMCachePodInjector is the mutating admission handler that wires an opted-in
// vLLM pod to an LMCacheEngine so the user no longer has to hand-write the
// --kv-transfer-config flag, hostIPC, and PYTHONHASHSEED. It is gated by the
// engine's connection ConfigMap: it mutates a pod only when the pod's
// lmcache.ai/lmcache-engine annotation names an engine whose <engine>-connection
// ConfigMap exists. It fails open (failurePolicy: Ignore) and is idempotent.
//
// It additionally reads the LMCacheEngine CR for its optional injection sub-spec:
// when injection.payloadImage is set it stages that lmcache code payload into the
// vLLM container (emptyDir + init container + read-only mount + PYTHONPATH),
// mirroring the CacheBlend injector. The connection wiring always runs; the
// payload staging is additive and skipped when no CR / payloadImage is present.
type LMCachePodInjector struct {
	// Client reads the engine's connection ConfigMap and the LMCacheEngine CR. It
	// uses the shared manager ServiceAccount, whose RBAC already grants
	// configmaps get and lmcacheengines get.
	Client client.Client

	// Decoder decodes the admission request's raw pod object.
	Decoder admission.Decoder
}

// Handle implements admission.Handler. It applies the LMCache connection
// mutations (hostIPC, --kv-transfer-config, PYTHONHASHSEED) to an opted-in pod
// whose named LMCacheEngine connection ConfigMap exists, optionally followed by
// code-payload staging when the engine's injection.payloadImage is set, then
// returns a JSON patch. It short-circuits to an unchanged Allowed response for
// non-opted-in or already-injected pods, and stamps a skip-reason annotation
// (still Allowed, fail-open) when it declines to mutate (engine missing, command
// override, target container missing, or user-supplied --kv-transfer-config).
func (p *LMCachePodInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	log := ctrl.LoggerFrom(ctx)

	pod, engineName, namespace, resp, handled := lmCacheKeys.gate(p.Decoder, req)
	if handled {
		return resp
	}

	// Read the engine CR for its optional injection defaults (payload staging +
	// target-container default). Fail open: a missing CR or absent injection
	// sub-spec just means connection-only injection — the connection ConfigMap
	// (read below) remains the real gate, so existing behavior is unchanged.
	engine := &lmcachev1alpha1.LMCacheEngine{}
	var targetDefault *string
	if err := p.Client.Get(ctx, types.NamespacedName{Name: engineName, Namespace: namespace}, engine); err != nil {
		if !apierrors.IsNotFound(err) {
			return admission.Errored(http.StatusInternalServerError, err)
		}
		// CR absent: engine.Spec.Injection stays nil → stage nothing, wire the
		// connection only (the connection ConfigMap below remains the gate).
	} else {
		engine.SetDefaults()
		if engine.Spec.Injection != nil {
			targetDefault = engine.Spec.Injection.TargetContainer
		}
	}

	// Read the connection ConfigMap (existence gate), resolve the target
	// container, and apply the command-override gate (shared with CacheBlend).
	kvTransferConfigJSON, containerIdx, resp, ok := prepareInjection(
		ctx, p.Client, req, pod, lmCacheKeys, engineName, namespace, targetDefault)
	if !ok {
		return resp
	}
	target := &pod.Spec.Containers[containerIdx]

	// user --kv-transfer-config gate: do not clobber the user's structured JSON.
	userHasKVTransferConfig := argsHasFlag(target.Args, lmcFlagKVTransferConfig)

	// --- Connection wiring (always applied) ---

	// M0: pod hostIPC for CUDA IPC with the node-local engine.
	pod.Spec.HostIPC = true

	// Args: inject --kv-transfer-config unless the user already supplied one.
	kvForArgs := kvTransferConfigJSON
	if userHasKVTransferConfig {
		kvForArgs = ""
	}
	target.Args = BuildLMCacheArgs(target.Args, kvForArgs)

	// Env: deterministic prefix hashing (set-if-absent).
	target.Env = BuildLMCacheEnv(target.Env)

	// --- Code-payload staging (only when injection.payloadImage is set) ---
	if engine.Spec.Injection != nil {
		stagePayload(ctx, pod, target, engine.Spec.Injection)
	}

	log.Info("Injected LMCache connection", "engine", engineName, "container", target.Name)
	return lmCacheKeys.stampInjected(req, pod, userHasKVTransferConfig)
}

// stagePayload applies the lmcache code-payload mutations to pod/target when the
// injection sub-spec resolves a non-empty payload image: a shared emptyDir, the
// payload init container, a read-only mount on the target vLLM container, a
// prepended PYTHONPATH, and the merged image-pull secrets (annotation override
// wins) for the private payload image. It is a no-op when payloadImage is unset,
// leaving the pod connection-wired only. All appends are idempotent within a
// single Handle call.
//
// Parameters:
//   - ctx: the request context (carries the logger).
//   - pod: the decoded pod (mutated in place: volumes, init containers, pull secrets).
//   - target: the resolved vLLM container (mutated in place: volume mount, env).
//   - injection: the engine's injection sub-spec (non-nil).
func stagePayload(
	ctx context.Context,
	pod *corev1.Pod,
	target *corev1.Container,
	injection *lmcachev1alpha1.LMCacheInjectionSpec,
) {
	payloadRef, payloadPullPolicy := resolvePayloadImage(injection.PayloadImage)
	if payloadRef == "" {
		return // No payload image configured: connection-only injection.
	}
	// NOTE: an empty payloadImage object (injection.payloadImage: {}) is NOT the
	// same as omitting it — the API server defaults repository to the shared
	// ImageSpec default (lmcache/vllm-openai), which is not a valid payload, so
	// staging it would run a non-payload image and stall the pod. This footgun is
	// inherited from the shared ImageSpec default and is identical for CacheBlend;
	// the field doc warns to set repository explicitly. Left unguarded to stay in
	// scope + in parity with CacheBlend (revisit if it materializes in practice).

	pod.Spec.Volumes = appendVolumeIfAbsent(pod.Spec.Volumes, lmcStaging.volume())
	pod.Spec.InitContainers = appendInitContainerIfAbsent(pod.Spec.InitContainers,
		lmcStaging.initContainer(payloadRef, payloadPullPolicy))
	target.VolumeMounts = appendVolumeMountIfAbsent(target.VolumeMounts, lmcStaging.volumeMount())
	target.Env = lmcStaging.prependPythonPath(target.Env)

	injectedSecrets := resolveInjectedPullSecrets(injection.ImagePullSecrets,
		pod.Annotations[LMCacheAnnotationImagePullSecrets])
	pod.Spec.ImagePullSecrets = MergeImagePullSecrets(pod.Spec.ImagePullSecrets, injectedSecrets)

	ctrl.LoggerFrom(ctx).Info("Staged LMCache payload", "image", payloadRef, "container", target.Name)
}
